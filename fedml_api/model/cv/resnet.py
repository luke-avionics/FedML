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


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     # padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
    # """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
    # expansion = 1

    # def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 # base_width=64, dilation=1, norm_layer=None):
        # super(BasicBlock, self).__init__()
        # if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
            # raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
            # raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        # self.downsample = downsample
        # self.stride = stride

    # def forward(self, x):
        # identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        # if self.downsample is not None:
            # identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # return out


# class Bottleneck(nn.Module):
    # expansion = 4

    # def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 # base_width=64, dilation=1, norm_layer=None):
        # super(Bottleneck, self).__init__()
        # if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        # self.stride = stride

    # def forward(self, x):
        # identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        # if self.downsample is not None:
            # identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # return out


# class ResNet(nn.Module):

    # def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 # width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        # super(ResNet, self).__init__()
        # if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
        # self._norm_layer = norm_layer

        # self.inplanes = 16
        # self.dilation = 1
        # if replace_stride_with_dilation is None:
            # # each element in the tuple indicates if we should replace
            # # the 2x2 stride with a dilated convolution instead
            # replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
            # raise ValueError("replace_stride_with_dilation should be None "
                             # "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # self.groups = groups
        # self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               # bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # # self.maxpool = nn.MaxPool2d()
        # self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.KD = KD
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
            # for m in self.modules():
                # if isinstance(m, Bottleneck):
                    # nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                    # nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # norm_layer = self._norm_layer
        # downsample = None
        # previous_dilation = self.dilation
        # if dilate:
            # self.dilation *= stride
            # stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                # norm_layer(planes * block.expansion),
            # )

        # layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            # self.base_width, previous_dilation, norm_layer))
        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
            # layers.append(block(self.inplanes, planes, groups=self.groups,
                                # base_width=self.base_width, dilation=self.dilation,
                                # norm_layer=norm_layer))

        # return nn.Sequential(*layers)

    # def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.layer1(x)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        # x = self.avgpool(x)  # B x 64 x 1 x 1
        # x_f = x.view(x.size(0), -1)  # B x 64
        # x = self.fc(x_f)  # B x num_classes
        # if self.KD == True:
            # return x_f, x
        # else:
            # return x


# def resnet56(class_num, pretrained=False, path=None, **kwargs):
    # """
    # Constructs a ResNet-110 model.

    # Args:
        # pretrained (bool): If True, returns a model pre-trained.
    # """
    # model = ResNet(Bottleneck, [6, 6, 6], class_num, **kwargs)
    # if pretrained:
        # checkpoint = torch.load(path)
        # state_dict = checkpoint['state_dict']

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
            # # name = k[7:]  # remove 'module.' of dataparallel
            # name = k.replace("module.", "")
            # new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
    # return model


# def resnet110(class_num, pretrained=False, path=None, **kwargs):
    # """
    # Constructs a ResNet-110 model.

    # Args:
        # pretrained (bool): If True, returns a model pre-trained.
    # """
    # logging.info("path = " + str(path))
    # model = ResNet(Bottleneck, [12, 12, 12], class_num, **kwargs)
    # if pretrained:
        # checkpoint = torch.load(path)
        # state_dict = checkpoint['state_dict']

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
            # # name = k[7:]  # remove 'module.' of dataparallel
            # name = k.replace("module.", "")
            # new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
    # return model
    
    
    
    
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
# ResNet-38
def cifar10_resnet_38(class_num,pretrained=False,path=None, **kwargs):
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
def cifar10_resnet_56(class_num,pretrained=False,path=None, **kwargs):
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
def cifar10_resnet_74(class_num,pretrained=False,path=None, **kwargs):
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
def cifar10_resnet_110(class_num,pretrained=False,path=None, **kwargs):
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
def cifar10_resnet_152(class_num,pretrained=False,path=None, **kwargs):
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
