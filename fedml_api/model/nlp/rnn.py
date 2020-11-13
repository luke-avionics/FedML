import torch.nn as nn
import sys
import logging
import torch
import torch.nn.functional as F
from .quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN


ACT_FW = 0
ACT_BW = 0
GRAD_ACT_ERROR = 0
GRAD_ACT_GC = 0

MOMENTUM = 0.9

DWS_BITS = 8
DWS_GRAD_BITS = 16

def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)
def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = QLinear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = _LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

class _LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, **kwargs):
        super().__init__()
        # self.input_projection = QConv2d(input_size, 4 * hidden_size, kernel_size=1, bias=bias)
        # self.hidden_projection = QConv2d(hidden_size, 4 * hidden_size, kernel_size=1, bias=bias)
        self.input_projection = QLinear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_projection = QLinear(hidden_size, 4 * hidden_size, bias=bias)
        self.hidden_size=hidden_size
        # self.ln_ii = nn.LayerNorm((hidden_size, ))
        # self.ln_hi = nn.LayerNorm((hidden_size, ))
        # self.ln_if = nn.LayerNorm((hidden_size, ))
        # self.ln_hf = nn.LayerNorm((hidden_size, ))
        # self.ln_ig = nn.LayerNorm((hidden_size, ))
        # self.ln_hg = nn.LayerNorm((hidden_size, ))
        # self.ln_io = nn.LayerNorm((hidden_size, ))
        # self.ln_ho = nn.LayerNorm((hidden_size, ))

    def forward(self, num_bits, num_bits_grad, input, hx=None):
        batch, embed_dim = input.size()
        hidden_dim=self.hidden_size
        h, c = hx

        # extended_input = input[:,:,None,None]
        # extended_hidden = h[:,:,None,None]

        mixed_input = self.input_projection(input, num_bits, num_bits_grad)
        mixed_hidden = self.hidden_projection(h, num_bits, num_bits_grad)
        
        mixed_input = mixed_input.squeeze().view((batch, 4, hidden_dim))
        mixed_hidden = mixed_hidden.squeeze().view((batch, 4, hidden_dim))

        i = torch.sigmoid(mixed_input[:, 0, :] + mixed_hidden[:, 0, :])
        f = torch.sigmoid(mixed_input[:, 1, :] + mixed_hidden[:, 1, :])
        g = torch.tanh(mixed_input[:, 2, :] + mixed_hidden[:, 2, :])
        o = torch.sigmoid(mixed_input[:, 3, :] + mixed_hidden[:, 3, :])

        # i = torch.sigmoid(self.ln_ii(mixed_input[:, 0, :]) + self.ln_hi(mixed_hidden[:, 0, :]))
        # f = torch.sigmoid(self.ln_if(mixed_input[:, 1, :]) + self.ln_hf(mixed_hidden[:, 1, :]))
        # g = torch.tanh(self.ln_ig(mixed_input[:, 2, :]) + self.ln_hg(mixed_hidden[:, 2, :]))
        # o = torch.sigmoid(self.ln_io(mixed_input[:, 3, :]) + self.ln_ho(mixed_hidden[:, 3, :]))
        
        c_ = f * c + i * g
        h_ = o * torch.tanh(c_)

        # print('============')
        # print(input)
        # print(h_)
        # print('============')
        
        
        return h_, c_

class LSTMDecoder(nn.Module):
    """LSTM decoder."""
    def __init__(
        self, vocab_size, embed_dim=512, hidden_size=512, out_embed_dim=512, padding_idx=0,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=False,
        encoder_output_units=0, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None
    ):
        super().__init__()
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.adaptive_softmax = None
        num_embeddings = vocab_size
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 
        self.layers = nn.ModuleList([
            LSTMCell( # TODO
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, num_bits, num_bits_grad, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x = self.extract_features(
            num_bits, num_bits_grad, prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x)

    def extract_features(
        self, num_bits, num_bits_grad, prev_output_tokens, encoder_out=None, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        encoder_padding_mask = None
        encoder_out = None
        #input
        bsz, seqlen = prev_output_tokens.size()

        srclen = None

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        #cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        #if cached_state is not None:
        #    prev_hiddens, prev_cells, input_feed = cached_state

        # setup zero cells, since there is no encoder
        num_layers = len(self.layers)
        zero_state = x.new_zeros(bsz, self.hidden_size)
        # uniform_state = zero_state.uniform_()
        prev_hiddens = [zero_state for i in range(num_layers)]
        prev_cells = [zero_state for i in range(num_layers)]
        input_feed = None


        outs = []
        for j in range(seqlen):
            input = x[j]
            for i, rnn in enumerate(self.layers):
                # recurrent cell # TODO
                hidden, cell = rnn(num_bits, num_bits_grad, input, (prev_hiddens[i], prev_cells[i]))
                # hidden state becomes the input to the next layer
                # input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                input = hidden
                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        # utils.set_incremental_state(
        #     self, incremental_state, 'cached_state',
        #     (prev_hiddens, prev_cells, input_feed),
        # )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        attn_scores = None
        return x

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x



def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


# def Embedding(num_embeddings, embedding_dim, padding_idx):
#     m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
#     nn.init.uniform_(m.weight, -0.1, 0.1)
#     nn.init.constant_(m.weight[padding_idx], 0)
#     return m

# def LSTMCell(input_size, hidden_size, **kwargs):
#   m = _LSTMCell(input_size, hidden_size, **kwargs)
#   for name, param in m.named_parameters():
#       if 'weight' in name or 'bias' in name:
#           param.data.uniform_(-0.1, 0.1)
#   return m

# class _LSTMCell(nn.Module):
#     def __init__(self, input_size, hidden_size, bias=True, **kwargs):
#         super().__init__()
#         # self.input_projection = QConv2d(input_size, 4 * hidden_size, kernel_size=1, bias=bias)
#         # self.hidden_projection = QConv2d(hidden_size, 4 * hidden_size, kernel_size=1, bias=bias)
#         self.input_projection = QLinear(input_size, 4 * hidden_size, bias=bias)
#         self.hidden_projection = QLinear(hidden_size, 4 * hidden_size, bias=bias)
#         self.input_projection2 = QLinear(hidden_size, 4 * hidden_size, bias=bias)
#         self.hidden_projection2 = QLinear(hidden_size, 4 * hidden_size, bias=bias)
#         self.hidden_size=hidden_size
#         self.input_size=input_size
#         # self.ln_ii = nn.LayerNorm((hidden_size, ))
#         # self.ln_hi = nn.LayerNorm((hidden_size, ))
#         # self.ln_if = nn.LayerNorm((hidden_size, ))
#         # self.ln_hf = nn.LayerNorm((hidden_size, ))
#         # self.ln_ig = nn.LayerNorm((hidden_size, ))
#         # self.ln_hg = nn.LayerNorm((hidden_size, ))
#         # self.ln_io = nn.LayerNorm((hidden_size, ))
#         # self.ln_ho = nn.LayerNorm((hidden_size, ))

#     def forward(self, num_bits, input, hx=None):
#         try:
#             #initial state
#             if hx is None:
#                 zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
#                 hx = (zeros, zeros,zeros,zeros)
#             batch, time_steps, input_size = input.size()
#             h1, c1, h2, c2 = hx
#         except Exception as e:
#             logging.info(str(e))
#             logging.info('ssss')
#             while(True):
#                 a=1

#         for time_step in range(time_steps):
#             try:
#                 mixed_input = self.input_projection(input[:,time_step,:].squeeze(dim=1), num_bits)
#                 mixed_hidden = self.hidden_projection(h1, num_bits)

#                 mixed_input = mixed_input.squeeze().view((batch, 4, self.hidden_size))
#                 mixed_hidden = mixed_hidden.squeeze().view((batch, 4, self.hidden_size))

#                 i = torch.sigmoid(mixed_input[:, 0, :] + mixed_hidden[:, 0, :])
#                 f = torch.sigmoid(mixed_input[:, 1, :] + mixed_hidden[:, 1, :])
#                 g = torch.tanh(mixed_input[:, 2, :] + mixed_hidden[:, 2, :])
#                 o = torch.sigmoid(mixed_input[:, 3, :] + mixed_hidden[:, 3, :])

#                 c1 = f * c1 + i * g
#                 h1 = o * torch.tanh(c1)

#                 mixed_input2 = self.input_projection2(h1,num_bits)
#                 mixed_hidden2 = self.hidden_projection2(h2,num_bits)

#                 mixed_input2 = mixed_input2.squeeze().view((batch, 4, self.hidden_size))
#                 mixed_hidden2 = mixed_hidden2.squeeze().view((batch, 4, self.hidden_size))

#                 i2 = torch.sigmoid(mixed_input2[:, 0, :] + mixed_hidden2[:, 0, :])
#                 f2 = torch.sigmoid(mixed_input2[:, 1, :] + mixed_hidden2[:, 1, :])
#                 g2 = torch.tanh(mixed_input2[:, 2, :] + mixed_hidden2[:, 2, :])
#                 o2 = torch.sigmoid(mixed_input2[:, 3, :] + mixed_hidden2[:, 3, :])

#                 c2 = f2 * c2 + i2 * g2
#                 h2 = o2 * torch.tanh(c2)
#             except Exception as e:
#                 exc_type, exc_obj, exc_tb = sys.exc_info()
#                 logging.info(str(e))
#                 logging.info(str(exc_tb.tb_lineno))
#         return h1, c1, h2, c2



class RNN_OriginalFedAvg(nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
      This replicates the model structure in the paper:
      Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629
      This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_OriginalFedAvg, self).__init__()
        #self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        #self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.lstm=LSTMDecoder(vocab_size=vocab_size, embed_dim=embedding_dim, hidden_size=hidden_size,
                              out_embed_dim=hidden_size,padding_idx=0, num_layers=2)
        #self.fc = Linear(hidden_size, vocab_size, bias=True)

    def forward(self, input_seq, num_bits=0):
        #embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out = self.lstm(num_bits, 0, input_seq)
        #lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        #final_hidden_state = lstm_out[:,-1,:].squeeze(1)
        #output = self.fc(final_hidden_state, num_bits)
        return lstm_out[:,-1,:].squeeze(1)


class RNN_StackOverFlow(nn.Module):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).
      This replicates the model structure in the paper:
      "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Table 9
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, vocab_size=10000,
                 num_oov_buckets=1,
                 embedding_size=96,
                 latent_size=670,
                 num_layers=1):
        super(RNN_StackOverFlow, self).__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size, embedding_dim=embedding_size,
                                            padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers)
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    def forward(self, input_seq, hidden_state = None):
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out[:,-1])
        output = self.fc2(fc1_output)
        return output
