import torch
import torch.nn as nn
from .quantize import QConv2d, RangeBN, QLinear


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
                   padding=1, bias=True, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
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

def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)

class MLP_fedcom(torch.nn.Module):
    def __init__(self):
        super(MLP_fedcom, self).__init__()
        self.linear_1 = Linear(32*32*3, 500)
        self.linear_2 = Linear(500, 500)
        self.linear_3 = Linear(500, 500)
        self.linear_4 = Linear(500, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    def forward(self, x, num_bits=0):
        x = torch.unsqueeze(x, 1)
        x = self.flatten(x)
        x = self.linear_1(x, num_bits)
        x = self.relu(x)
        x = self.linear_2(x, num_bits)
        x = self.relu(x)
        x = self.linear_3(x, num_bits)
        x = self.relu(x)
        x = self.linear_4(x, num_bits)