import logging
import copy
import torch
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list
from fedml_api.model.cv.quantize import calculate_qparams, quantize

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
        self.quant_residue = dict()
        self.first_run= True
        self.glb_epoch=0
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            if self.args.dataset =="femnist":
                self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr)
            elif self.args.dataset == "cifar10_fedcom":
                self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.args.lr, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)
        lr_steps = self.args.comm_round *self.args.epochs* len(self.train_local)
        if self.args.dataset =="femnist":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=1)
        elif self.args.dataset == "cifar10_fedcom":
            self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer,0.99)
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.5)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps / 2], gamma=0.1)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps * 3 / 4], gamma=0.5)
        self.comm_round = 0 
        logging.info("=======num_iter====== "+str(lr_steps))
        self.cyclic_period = self.args.comm_round
        self.lr_steps=lr_steps
    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.weights_t = weights
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):
        if self.args.client_num_in_total == self.args.client_num_per_round:
            #apply residue
            weights = self.model.cpu().state_dict()
            for k in self.quant_residue.keys():
                #if 'bias' in k:
                #    logging.info(str(k))
                if 'weight' in k and 'bn' not in k:
                    weights[k]=copy.deepcopy(weights[k]+self.quant_residue[k])
                elif 'bias' in k and 'bn' not in k:
                    weights[k]=copy.deepcopy(weights[k]+self.quant_residue[k])
            self.update_model(weights)
        else:
            logging.info('residue compensation skipped')
        #logging.info('Before training starts..............')
        self.model.to(self.device)
        # change to train mode
        self.model.train()
        #logging.info('model init done !!!!!!!!!!!!!')
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            try:
                for batch_idx, (x, labels) in enumerate(self.train_local):
                    
                    #logging.info('Beginning of training on one batch !!!!!!!!!!!!!!!!!!!!!!!!!!')

                    _iters = self.glb_epoch * len(self.train_local) + batch_idx
                    #cyclic_period = int((self.args.comm_round * self.args.epochs* len(self.train_local)) // self.cyclic_period)
                    cyclic_period = int((self.args.comm_round * self.args.epochs *  len(self.train_local)) // 16)
                    #logging.info("cyclic period: "+ str(cyclic_period))
                    #cyclic_period = int((self.args.comm_round * self.args.epochs *  len(self.train_local)) // self.cyclic_period)*2
                    #if self.glb_epoch % 2 == 0:
                    #    self.args.cyclic_num_bits_schedule=[8,32]
                    #else:
                    #    self.args.cyclic_num_bits_schedule=[4,32]

                    if (self.args.cyclic_num_bits_schedule[0]==0 or self.args.cyclic_num_bits_schedule[1]==0) :
                        num_bits = 0
                    #elif self.glb_epoch>=self.args.comm_round-3:
                    #    num_bits = 8
                    # elif self.glb_epoch>=self.args.comm_round-10:
                    #     #self.args.inference_bits=32
                    #     self.args.cyclic_num_bits_schedule=[4,32]
                    #     cyclic_period = int((self.args.comm_round * len(self.train_local)) // 64)
                    #     offset=self.offset_finder(self.args.cyclic_num_bits_schedule[1],cyclic_period,len(self.train_local),self.lr_steps)
                    #     offseted_iters=min(max(0,_iters-offset),self.lr_steps)
                    #     num_bits = self.cyclic_adjust_precision(offseted_iters, cyclic_period)
                    else:
                        offset=self.offset_finder(self.args.cyclic_num_bits_schedule[1],cyclic_period,len(self.train_local),self.lr_steps)
                        offseted_iters=min(max(0,_iters-offset),self.lr_steps)
                        num_bits = self.cyclic_adjust_precision(offseted_iters, cyclic_period)
                        # num_bits = self.cyclic_adjust_precision(_iters, cyclic_period)
                        #if num_bits == 32:
                        #    num_bits =0
                    #logging.info('Right before data moving starts!!!!!')
                    # logging.info(images.shape)
                    x, labels = x.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    # if epoch < 10 and  self.first_run:
                    #     log_probs = self.model(x, num_bits=0)
                    # else:
                    #logging.info('Right before training started !!!!!!!!!!!!!!!!!')
                    log_probs = self.model(x, num_bits=num_bits)
                    loss = self.criterion(log_probs, labels)
                    l2_norm = 0
                    weights_local = self.model.named_parameters()
                    for key in self.weights_t:
                        if key in weights_local:
                            l2_norm += torch.norm(weights_local[key] - weights_t[key])
                    loss += 0.01 * l2_norm
                    loss.backward()
                    if self.args.client_num_in_total == self.args.client_num_per_round:
                        g_norm=nn.utils.clip_grad_norm_(self.model.parameters(),0.9,'inf')
                    else:
                        g_norm=nn.utils.clip_grad_norm_(self.model.parameters(),0.9,'inf')
                    #logging.info(str(g_norm))
                    self.optimizer.step()
                    batch_loss.append(loss.item())

                    self.scheduler.step()

                    #logging.info('End of training on one batch !!!!!!!!!!!!!!!!!!!!!!!!!!')

                self.glb_epoch+=1
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                    epoch, sum(epoch_loss) / len(epoch_loss)))
            except Exception as e:
                logging.info(str(e))
        # if self.comm_round < (self.args.lr_decay_step_size+1):
        #     self.scheduler.step()
        self.comm_round+=1
        for g in self.optimizer.param_groups:
            logging.info("===current learning rate===: "+str(g['lr']))
            break
        logging.info("========= number of batches =======: "+str(batch_idx+1))
        #logging.info("========= Transmitted bits ========: "+str(num_bits))
        self.first_run=False

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        latent_weight=copy.deepcopy(weights)
        logging.info('Quantizing model')
        if num_bits != 0:
            for k in weights.keys():
                #if 'bias' in k:
                #    logging.info(str(k))
                if 'weight' in k and 'bn' not in k:
                    weight_qparams = calculate_qparams(copy.deepcopy(weights[k]), num_bits=num_bits, flatten_dims=(1, -1),
                                                    reduce_dim=None)
                    weights[k] = quantize(copy.deepcopy(weights[k]), qparams=weight_qparams)
                    self.quant_residue[k]=copy.deepcopy(latent_weight[k]-weights[k])
                elif 'bias' in k and 'bn' not in k:
                    weights[k] = quantize(copy.deepcopy(weights[k]), num_bits=num_bits,flatten_dims=(0, -1))
                    self.quant_residue[k]=copy.deepcopy(latent_weight[k]-weights[k])
        logging.info('Sending model')
        return weights, self.local_sample_number, num_bits, latent_weight

    def cyclic_adjust_precision(self, _iters, cyclic_period, fixed_sch=True,print_bits=True):
        if self.args.cyclic_num_bits_schedule[0]==self.args.cyclic_num_bits_schedule[1]:
            return self.args.cyclic_num_bits_schedule[0]

        up_period = cyclic_period / 2
        down_period = cyclic_period / 2
        current_iter = _iters % cyclic_period

        num_bit_min = self.args.cyclic_num_bits_schedule[0]
        num_bit_max = self.args.cyclic_num_bits_schedule[1]

        if current_iter < up_period:
            slope = float(num_bit_max - num_bit_min)/up_period
            num_bits = round(slope * current_iter)+num_bit_min
        else:
            slope = - float(num_bit_max - num_bit_min)/down_period
            num_bits = round(slope * (current_iter-up_period)) + num_bit_max
        if print_bits:
            if _iters % 50 == 0:
                logging.info('num_bits = {} '.format(num_bits))

        return num_bits

    def offset_finder(self, max_bit, cyclic_period, batch_num,total_iter):
        for i in range(0,total_iter-1):
            last=self.cyclic_adjust_precision(i, cyclic_period, print_bits=False)
            new=self.cyclic_adjust_precision(i+1,cyclic_period, print_bits=False)
            if last==max_bit and new==max_bit-1:
                #print(i)
                break
        offset=0
        for j in range(0,batch_num):
            if (i-j) % batch_num ==0:
                offset=-j
                break
            elif (i+j) % batch_num ==0:
                offset=j
                break
        return offset
