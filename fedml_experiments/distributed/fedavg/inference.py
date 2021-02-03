import argparse
import logging
import os
import random
import socket
import sys
from torch import nn
import time
import numpy as np
import psutil
import setproctitle
import torch
import wandb
import copy
import logging

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.model.cv.resnet import resnet20, resnet38, resnet74, resnet110
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.distributed.fedavg.utils import transform_list_to_tensor
from fedml_api.model.cv.quantize import calculate_qparams, quantize


class model_inf():
    def __init__(self, ckpt_file,inference_bits):

        self.dataset='cifar10'
        self.data_dir='/home/yf22/dataset/'
        self.partition_alpha=0.5
        self.partition_method='hetero'
        self.client_num_in_total=1
        self.batch_size=32
        self.output_dim=10
        self.inference_bits=inference_bits
        self.ckpt_file=ckpt_file
        self.model = resnet20(class_num=self.output_dim)
        self.train_data_num, self.test_data_num, self.train_data_global, self.test_data_global, \
        self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict, \
        self.class_num = load_partition_data_cifar10(self.dataset, self.data_dir, self.partition_method,
                                self.partition_alpha, self.client_num_in_total, self.batch_size)


    def _infer(self, test_data):
        self.model.load_state_dict(torch.load(self.ckpt_file))
        self.model.eval()
        self.model.cuda()

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.cuda()
                target = target.cuda()
                pred = self.model(x, num_bits=self.inference_bits)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
        #torch.save(self.model.state_dict(), '/home/yz87/FedML/fedml_experiments/distributed/fedavg/baseline_non_iid_1.ckpt')
        return test_acc, test_total, test_loss


model_inf1=model_inf("",0)
test_num_samples = []
test_tot_corrects = []
test_losses = []

test_tot_correct, test_num_sample, test_loss = model_inf1._infer(model_inf1.test_data_global)
#test_tot_correct, test_num_sample, test_loss = self._infer_test(self.test_data_local_dict[client_idx],idx)
test_tot_corrects.append(copy.deepcopy(test_tot_correct))
test_num_samples.append(copy.deepcopy(test_num_sample))
test_losses.append(copy.deepcopy(test_loss))
test_acc = sum(test_tot_corrects) / sum(test_num_samples)
test_loss = sum(test_losses) / sum(test_num_samples)
print(test_acc)
print(test_loss)