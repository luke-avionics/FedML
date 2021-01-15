import copy
import logging
import time

import torch
import wandb
import numpy as np
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor


class FedAVGAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)

    def init_model(self, model):
        model_params = model.state_dict()
        # logging.info(model)
        return model, model_params

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True
    def quantize_grad(self, grad,bits):
        """quantize the tensor grad in s level on the absolute value coef wise"""
        norm=torch.norm(grad)
        #s=torch.floor(2**(bits-1)/(torch.max(grad)/norm))
        s=2.0**bits
        level_float = s * torch.abs(grad) / norm
        previous_level = torch.floor(level_float)
        is_next_level = torch.rand(*grad.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        if bits == 1:
            return torch.sign(grad) * norm / s 
        else:
            return torch.sign(grad) * norm * new_level / s

    def aggregate(self,previous_global_model_params):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        # for k in averaged_params.keys():
        #     for i in range(0, len(model_list)):
        #         local_sample_number, local_model_params = model_list[i]
        #         w = local_sample_number / training_num
        #         if i == 0:
        #             averaged_params[k] = local_model_params[k] * w
        #         else:
        #             averaged_params[k] += local_model_params[k] * w
        for k in averaged_params.keys():
            if  ('running_var' not in k) and ('running_mean' not in k) and ('num_batches_tracked' not in k):
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        #quantize gradient here
                        if self.args.grad_bits is None or self.args.grad_bits==0:
                            #no quantization
                            averaged_params[k] = (local_model_params[k]-previous_global_model_params[k]) * w + previous_global_model_params[k]
                        else: 
                            averaged_params[k] = self.quantize_grad(local_model_params[k]-previous_global_model_params[k], self.args.grad_bits) * w + previous_global_model_params[k]
                    else:
                        if self.args.grad_bits is None or self.args.grad_bits==0:
                            #no quantization
                            averaged_params[k] += (local_model_params[k]-previous_global_model_params[k]) * w  
                        else:
                            averaged_params[k] += self.quantize_grad(local_model_params[k]-previous_global_model_params[k], self.args.grad_bits) * w  
            else:
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.model.load_state_dict(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_all_clients(self, round_idx,traffic_count,client_indexes):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            test_num_samples = []
            test_tot_corrects = []
            test_losses = []
            #tmp_glb_dict=self.model.state_dict()
            for idx, client_idx in enumerate(client_indexes):
                # train data
                train_tot_correct, train_num_sample, train_loss = self._infer(self.train_data_local_dict[client_idx])
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # test data
                test_tot_correct, test_num_sample, test_loss = self._infer(self.test_data_local_dict[client_idx])
                #test_tot_correct, test_num_sample, test_loss = self._infer_test(self.test_data_local_dict[client_idx],idx)
                test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                test_num_samples.append(copy.deepcopy(test_num_sample))
                test_losses.append(copy.deepcopy(test_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break
            #self.model.load_state_dict(tmp_glb_dict)
            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx},commit=False)
            wandb.log({"Train/Loss": train_loss, "round": round_idx}, commit=False)
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx},commit=False)
            wandb.log({"Test/Loss": test_loss, "round": round_idx},commit=False)
            wandb.log({"Test/Acc": test_acc, "traffic_volume": traffic_count*2})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

    def _infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x, num_bits=self.args.inference_bits)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss

    def _infer_test(self, test_data, index):
        self.model.load_state_dict(self.model_dict[index])
        self.model.eval()
        self.model.to(self.device)
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x, num_bits=self.args.inference_bits)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss

