'''
ResNet for CIFAR-10/100 Dataset.
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
'''
import logging

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
__all__ = ['ResNet', 'cifar100_resnet_110']
   
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


bn_layer_dict = {}


def get_bn_layer(planes, identifier):
    # if identifier not in bn_layer_dict:
    #     bn_layer_dict[identifier] = nn.BatchNorm2d(planes, affine=True)
    # return bn_layer_dict[identifier]
    return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, identifier, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = get_bn_layer(planes, identifier + "1")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = get_bn_layer(planes, identifier + "2")
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################


class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = get_bn_layer(16, "1")
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], "layer1")
        self.layer2 = self._make_layer(block, 32, layers[1], "layer2", stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], "layer3", stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, identifier, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_bn_layer(planes * block.expansion, identifier),
            )

        layers = []
        layers.append(block(self.inplanes, planes, identifier + "0", stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, identifier + str(i)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# For CIFAR-10

# ResNet-20
def resnet20(class_num, pretrained=False, path=None, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [3, 3, 3], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
    
# ResNet-38
def resnet38(class_num,pretrained=False,path=None, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

# ResNet-56
def resnet56(class_num,pretrained=False,path=None, **kwargs):
    # n = 9
    model = ResNet(BasicBlock, [9, 9, 9], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

# ResNet-74
def resnet74(class_num,pretrained=False,path=None, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


# ResNet-110
def resnet110(class_num,pretrained=False,path=None, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


# ResNet-152
def resnet152(class_num,pretrained=False,path=None, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

'''
# For CIFAR-100
# ResNet-38
def resnet38(class_num,pretrained=False,path=None, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=class_num,**kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


# ResNet-74
def cifar100_resnet_74(class_num,pretrained=False,path=None, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=class_num,**kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


# ResNet-110
def cifar100_resnet_110(class_num,pretrained=False,path=None, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=class_num,**kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


# ResNet-152
def cifar100_resnet_152(class_num,pretrained=False,path=None, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=class_num,**kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
'''