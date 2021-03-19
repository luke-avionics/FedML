import logging
import math
import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list, hook_for_BNLoss
from fedml_api.model.cv.quantize import calculate_qparams, quantize
from fedml_api.model.cv.resnet import resnet20
from fedml_api.model.cv.cnn import CNNCifar

from fedml_api.distributed.fedavg.generator import Generator

class FedAVGTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 args):
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.indicator_use_share_data = None
        self.total_batch_num = len(self.train_local)

        self.device = device
        self.args = args
        self.model = model
        self.first_run= True
        self.glb_epoch=0
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.args.lr, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)
        lr_steps = self.args.comm_round * len(self.train_local)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.epochs * len(self.train_local), gamma=0.992)
        self.comm_round = 0 

        self.cyclic_period = self.args.comm_round
        self.lr_steps=lr_steps
    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index, shared_data=None):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

        if shared_data is not None:
            for item in shared_data:
                item = [item[0].to(self.device), item[1].to(self.device)]

            self.shared_data = shared_data  # a list with size batch_num: [((batch_size, channel_num, height, weight), label)]

            
            # logging.info("+++++++++++shared data: " + str(shared_data))
            self.total_batch_num = len(self.train_local) + len(shared_data)

            share_location = np.random.choice(self.total_batch_num, len(shared_data))

            self.indicator_use_share_data = np.zeros(self.total_batch_num)
            self.indicator_use_share_data[share_location] = 1
            logging.info('Using fake data')
            #logging.info('fake data indicator:' + str(self.indicator_use_share_data))
        else:
            logging.info('Not using fake data')
            self.total_batch_num = len(self.train_local)
            self.indicator_use_share_data = None
        

    def train(self):
        #logging.info('Before training starts..............')
        self.model.to(self.device)
        # change to train mode
        self.model.train()
        #logging.info('model init done !!!!!!!!!!!!!')

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            shared_data_idx = 0
            # try:
            for batch_idx in range(self.total_batch_num):
                try:    
                    #logging.info('Beginning of training on one batch !!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #logging.info('Right before data moving starts!!!!!')
                    # logging.info(images.shape)

                    if self.indicator_use_share_data is not None and self.indicator_use_share_data[batch_idx]:
                        x, labels = self.shared_data[shared_data_idx]
                        shared_data_idx += 1
                    else:
                        x, labels = next(iter(self.train_local))

                    x, labels = x.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    # if epoch < 10 and  self.first_run:
                    #     log_probs = self.model(x, num_bits=0)
                    # else:
                    #logging.info('Right before training started !!!!!!!!!!!!!!!!!')
                    log_probs = self.model(x)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    g_norm=nn.utils.clip_grad_norm_(self.model.parameters(),0.9,'inf')
                    #logging.info(str(g_norm))
                    self.optimizer.step()
                    batch_loss.append(loss.item())

                    self.scheduler.step()

                    #logging.info('End of training on one batch !!!!!!!!!!!!!!!!!!!!!!!!!!')
                except Exception as e:
                    logging.warning(str(e))
            self.glb_epoch+=1
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))
            # except Exception as e:
            #     logging.info(str(e))
        # if self.comm_round < (self.args.lr_decay_step_size+1):
        #     self.scheduler.step()
        self.comm_round+=1
        for g in self.optimizer.param_groups:
            #logging.info("===current learning rate===: "+str(g['lr']))
            break
        #logging.info("========= number of batches =======: "+str(batch_idx+1))
        # logging.info("========= Transmitted bits ========: "+str(num_bits))
        self.first_run=False

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        return weights, self.local_sample_number

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
        #if print_bits:
            #if _iters % 50 == 0:
                #logging.info('num_bits = {} '.format(num_bits))

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





class ServerTrainer(object):
    def __init__(self, device):
        self.device = device

        self.batch_num = 50

        self.epochs = 20
        self.iters = 500
        self.batch_size = 100
        self.latent_dim = 512
        self.alpha = 0.01

        self.model = resnet20(class_num = 10).to(self.device)

        self.generator = Generator(latent_dim=self.latent_dim, img_size=32).to(self.device)

        self.lr = 0.01
        self.lr_steps = 1000

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.optimizer_G = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr, weight_decay=1e-4)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=1e-3,betas=(0.5,0.999))

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.lr_steps / 2, self.lr_steps * 3 / 4], gamma=0.1)

        self.invoke_idx = 0

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)


    def generate_fake_data(self, global_model_params):
        
        logging.info('Central Server: Generating fake data..............')

        '''
        # self.model.to(self.device)
        # change to train mode
        self.update_model(global_model_params) # Error(s) in loading state_dict for ResNet unexpected key(s)
        # self.model.train()
        #logging.info('model init done !!!!!!!!!!!!!')

        self.model.eval() # teacher network/global model
        self.generator.train()


        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=200, eta_min=0)

        loss_r_feature_layers = []

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(hook_for_BNLoss(module))

        for epoch in range(self.epochs):
            for i in range(self.iters):
                z = torch.randn(self.batch_size, self.latent_dim).cuda()
                self.optimizer_G.zero_grad()

                gen_imgs = self.generator(z)

                o_T = self.model(gen_imgs)
                so_T = torch.nn.functional.softmax(o_T, dim = 1)
                so_T_mean=so_T.mean(dim = 0)

                l_ie = (so_T_mean * torch.log(so_T_mean)).sum() #IE loss
                l_oh= - (so_T * torch.log(so_T)).sum(dim=1).mean() #one-hot entropy
                l_bn = 0 #BN loss

                for mod in loss_r_feature_layers:
                    l_bn += mod.G_kd_loss.sum()  

                l_s = self.alpha * (l_ie + l_oh + l_bn)

                l_s.backward()
                self.optimizer_G.step()


            logging.info("Central Server: Fake Data Generation [Epoch %d/%d]  [loss_oh: %f] [loss_ie: %f] [loss_BN: %f] ",
                        epoch, self.epochs, l_oh.item(), l_ie.item(), l_bn.item())
            
            scheduler_G.step()

        gen_imgs_list = []
        labels_list = []
        for batch_id in range(self.batch_num):
            z = torch.randn(self.batch_size, self.latent_dim).cuda()
            gen_imgs = self.generator(z)  #cuda

            logits = self.model(gen_imgs)   #cuda
            labels = logits.argmax(-1)   #cuda
            
            
            
            
            class_num = len(torch.unique(labels))
            data_per_class = round(self.batch_size /class_num)    # number of fake data needed to choose for each class
            
            #print(class_num)
            #logging.info('(number of classes: {} '.format(class_num))
            #index_list = []
            final_gen_imgs = torch.empty([0, gen_imgs.size()[-3], gen_imgs.size()[-2], gen_imgs.size()[-1]]).cuda()
            final_labels = torch.empty([0], dtype=torch.long).cuda()
            
            
            for class_id in torch.unique(labels):
                index = (labels==class_id).nonzero().squeeze(dim=1)
                final_labels = torch.cat((final_labels, class_id*torch.ones(data_per_class, dtype=torch.long).cuda() ))
                if len(index) >= data_per_class:
                    temp = gen_imgs[index]   
                    final_gen_imgs = torch.cat((final_gen_imgs, temp[:data_per_class]))
                    
                else: 
                    temp = gen_imgs[index]  
                    temp = temp.repeat(math.ceil(data_per_class/len(index)), 1, 1, 1)
                    final_gen_imgs = torch.cat((final_gen_imgs, temp[:data_per_class]))     
                         
                
            gen_imgs_list.append(final_gen_imgs.to('cpu'))
            labels_list.append(labels.to('cpu'))
            
            
            gen_imgs_list.append(gen_imgs.to('cpu'))
            labels_list.append(labels.to('cpu'))

        shared_data = list(zip(gen_imgs_list, labels_list)) 
        
        # Save first 10 images in the first batch
        for i in range(10):
            save_image(shared_data[0][0][i], './sample_imgs/10client_cifar10/iter{}_image{}_label{}.png'.format(self.invoke_idx, i, shared_data[0][1][i]))

        # generate fake data
        #shared_data = [[np.ones((8, 3, 32, 32)), np.ones((8))] for _ in range(32)]
        # logging.info("{}".format(shared_data[0]))
        self.invoke_idx += 1       
        
        
        
        # Choice 2: import existing images as fake data
        from PIL import Image
        from fedml_api.data_preprocessing.cifar10.data_loader import Cutout
        import os 
        from random import sample,randint
        from torchvision import transforms
        import torchvision.transforms as transforms
        
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        loader = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])  
        
        loader.transforms.append(Cutout(16))
        
        gen_imgs_list = []
        labels_list = []
        
        num_data_per_class = int(self.batch_size/10)
        compl = self.batch_size % 10
        
        for _ in range(self.batch_num):
        
            gen_imgs = torch.empty([0, 3, 32, 32])
            labels = torch.empty([0], dtype=torch.long)
            
            for class_id in range(10):
                file_path = '/home/mz44/FedML/fedml_api/distributed/fedavg/gen_img/'+ str(class_id) + '/'
                file_name_list = os.listdir(file_path)
                selected_files = sample(file_name_list, num_data_per_class)
                for file_name in selected_files:
                    select_file = file_path + file_name
                    im = loader(Image.open(select_file)).unsqueeze(dim=0)
                    #im = np.transpose(im, (2,0,1))
                    gen_imgs = torch.cat((gen_imgs, im))
                    labels = torch.cat((labels, torch.tensor([class_id], dtype=torch.long)))
        
            selected_classes = sample(range(10), compl)
            for class_id in selected_classes:
                file_path = '/home/mz44/FedML/fedml_api/distributed/fedavg/gen_img/'+ str(class_id) + '/'
                file_name_list = os.listdir(file_path)
                selected_file = file_path + file_name_list[randint(0,len(file_name_list)-1)]
                im = loader(Image.open(selected_file)).unsqueeze(dim=0)
                gen_imgs = torch.cat((gen_imgs, im))
                labels = torch.cat((labels, torch.tensor([class_id], dtype=torch.long)))
            gen_imgs_list.append(gen_imgs)
            labels_list.append(labels)
        
        shared_data = list(zip(gen_imgs_list, labels_list))     
            # End of using existing fake images
          '''  
        
        # Use real images   
        # if the shared data is fixed, uncomment line 
        
        import argparse 
        #from PIL import Image
        #import os 
        from random import sample,randint
        from torchvision import transforms
        import torchvision.transforms as transforms
        from main_fedavg import add_args, load_data
        
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        
        parser = argparse.ArgumentParser()
        args = add_args(parser)
        dataset = load_data(args, args.dataset)
        [train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        train_set = train_data_global.dataset
        data= train_set.data    #ndarray (50000,32,32,3)
        label = train_set.target   # ndarray(50000,)
        loader = transforms.Compose([transforms.ToPILImage(), transforms.Scale(32), transforms.ToTensor(), ])  #transforms.Normalize(CIFAR_MEAN, CIFAR_STD), 
        
        gen_imgs_list = []
        labels_list = []          
        
        for batch_idx in range(self.batch_num):
        
            gen_imgs = torch.empty([0, 3, 32, 32])
            labels = torch.empty([0], dtype=torch.long)
           
            for i in range(self.batch_size):
                #idx = randint(0, data.shape[0]-1)
                ####################################################################
                idx = batch_idx * self.batch_size + i + 40000 
                ####################################################################
                im = loader(data[idx ,:,:,:]).unsqueeze(dim=0)   # (1,3,32,32)
               
                gen_imgs = torch.cat((gen_imgs, im))
                labels = torch.cat((labels, torch.tensor([label[idx]], dtype=torch.long)))
            
            
            gen_imgs_list.append(gen_imgs)
            labels_list.append(labels)
          # end of using real data
       
        shared_data = list(zip(gen_imgs_list, labels_list))     
            
        return shared_data   # a list with size batch_num: [((batch_size, channel_num, height, weight), label)]
