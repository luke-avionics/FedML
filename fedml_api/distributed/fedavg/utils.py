import torch
import numpy as np
import logging


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

class hook_for_BNLoss():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        T_mean = module.running_mean.data
        T_var = module.running_var.data
        self.G_kd_loss = self.Gaussian_kd(mean, var, T_mean, T_var)

    def Gaussian_kd(self, mean, var, T_mean, T_var):

        num = (mean-T_mean)**2 + var
        denom = 2*T_var
        std = torch.sqrt(var)
        T_std = torch.sqrt(T_var)

        return num/denom - torch.log(std/T_std) - 0.5

    def close(self):
        self.hook.remove()