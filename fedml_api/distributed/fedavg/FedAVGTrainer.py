import logging

import torch
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list


class FedAVGTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 args):
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):
        if self.args.cyclic_num_bits_schedule is None:
            num_bits = 0
        else:
            num_bits = self.cyclic_adjust_precision(epoch)

        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if epoch < 10 and  round_idx ==0:
                    log_probs = self.model(x, num_bits=0)
                else:
                    log_probs = self.model(x, num_bits=num_bits)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def cyclic_adjust_precision(self, epoch, fixed_sch=True):
        if self.args.cyclic_num_bits_schedule[0]==self.args.cyclic_num_bits_schedule[1]:
            return self.args.cyclic_num_bits_schedule[0]


        if self.args.epochs % 20==0:
            if self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==8:
                #[4-8]
                sch=[4,6,7,8,7,6,4,6,7,8,8,7,6,4,6,7,8,7,6,4]
            elif self.args.cyclic_num_bits_schedule[0]==8 and self.args.cyclic_num_bits_schedule[1]==32:
                #[8-32]
                sch=[8,16,24,32,32,24,16,8,16,24,32,24,16,8,16,24,32,24,16,8]
            elif self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==16:
                #[4-16]
                sch=[4,8,12,16,16,12,8,4,8,12,16,12,8,4,8,12,16,12,8,4]
            elif self.args.cyclic_num_bits_schedule[0]==8 and self.args.cyclic_num_bits_schedule[1]==16:
                #[8-16]
                sch=[8,10,13,16,16,13,10,8,10,13,16,13,10,8,10,13,16,13,10,8]
            elif self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==32:
                #[4-32]
                sch=[4,13,22,32,32,22,13,4,13,22,32,22,13,4,13,22,32,22,13,4]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==8:
                #[2-8]
                sch=[2,4,6,8,8,6,4,2,4,6,8,6,4,2,4,6,8,6,4,2]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==16:
                #[2-16]
                sch=[2,6,11,16,16,11,6,2,6,11,16,11,6,2,6,11,16,11,6,2]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==32:
                #[2-32]
                sch=[2,12,22,32,32,22,12,2,12,22,32,22,12,2,12,22,32,22,12,2]
            elif self.args.cyclic_num_bits_schedule[0]==3 and self.args.cyclic_num_bits_schedule[1]==8:
                sch=[3, 4, 5, 6, 8, 8, 6, 5, 4, 3, 3, 4, 5, 6, 8, 8, 6, 5, 4, 3]
        else:
            if self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==8:
                #[4-8]
                sch=[4, 5, 6, 7, 8, 8, 7, 6, 5, 4]
            elif self.args.cyclic_num_bits_schedule[0]==8 and self.args.cyclic_num_bits_schedule[1]==32:
                #[8-32]
                sch=[8, 14, 20, 26, 32, 32, 26, 20, 14, 8]
            elif self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==16:
                #[4-16]
                sch=[4, 7, 10, 13, 16, 16, 13, 10, 7, 4]
            elif self.args.cyclic_num_bits_schedule[0]==8 and self.args.cyclic_num_bits_schedule[1]==16:
                #[8-16]
                sch=[8, 10, 12, 14, 16, 16, 14, 12, 10, 8]
            elif self.args.cyclic_num_bits_schedule[0]==4 and self.args.cyclic_num_bits_schedule[1]==32:
                #[4-32]
                sch=[4, 11, 18, 25, 32, 32, 25, 18, 11, 4]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==8:
                #[2-8]
                sch=[2, 3, 5, 6, 8, 8, 6, 5, 3, 2]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==16:
                #[2-16]
                sch=[2, 5, 9, 12, 16, 16, 12, 9, 5, 2]
            elif self.args.cyclic_num_bits_schedule[0]==2 and self.args.cyclic_num_bits_schedule[1]==32:
                #[2-32]
                sch=[2, 9, 17, 24, 32, 32, 24, 17, 9, 2]

        #[4-8]
        #sch=[4,6,7,8,7,6,4,6,7,8,8,7,6,4,6,7,8,7,6,4]
        #[8-32]
        #sch=[8,16,24,32,32,24,16,8,16,24,32,24,16,8,16,24,32,24,16,8]
        if not fixed_sch:
            up_period = math.ceil(self.args.epochs / 2 * self.args.cyclic_period)
            down_period =  math.ceil(self.args.epochs / 2 * self.args.cyclic_period)
            current_iter = epoch % self.args.epochs

            num_bit_min = self.args.cyclic_num_bits_schedule[0]
            num_bit_max = self.args.cyclic_num_bits_schedule[1]

            if current_iter%(up_period+down_period) < up_period:
                slope = float(num_bit_max - num_bit_min)/up_period
                num_bits = round(slope * (current_iter%(up_period+down_period)+1))+num_bit_min
            #elif current_iter == (self.args.epochs-1):
            #    num_bits = num_bit_min
            else:
                slope = - float(num_bit_max - num_bit_min) / down_period
                num_bits = round(slope * ((current_iter%(up_period+down_period)+1)-up_period)) + num_bit_max
        else:
            num_bits=sch[epoch % 20]
        logging.info('Local epoch [{}] num_bits = {} '.format(epoch, num_bits))

        return num_bits
