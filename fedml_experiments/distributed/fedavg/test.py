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
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10,calculate_emd
from fedml_api.distributed.fedavg.utils import transform_list_to_tensor
from fedml_api.model.cv.quantize import calculate_qparams, quantize


class EMD():
    def __init__(self):

        self.dataset='cifar10'
        self.data_dir='/home/yf22/dataset/'
        self.partition_alpha=0.2
        self.partition_method='hetero'
        self.client_num_in_total=10
        self.batch_size=32
        self.output_dim=8
        self.train_data_num, self.test_data_num, self.train_data_global, self.test_data_global, \
        self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict, \
        self.class_num = load_partition_data_cifar10(self.dataset, self.data_dir, self.partition_method,
                                self.partition_alpha, self.client_num_in_total, self.batch_size)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
emd1=EMD()


calculate_emd(emd1.train_data_num,emd1.train_data_local_num_dict,emd1.train_data_local_dict,"/home/yf22/dataset")
