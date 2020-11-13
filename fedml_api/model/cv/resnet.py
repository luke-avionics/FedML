""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
from .quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import torch.nn.functional as F


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
                   padding=1, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)


def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = make_bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = make_bn(planes)
        self.bn3 = make_bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits=0):
        residual = x

        out = self.conv1(x, num_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x, num_bits)
            residual = self.bn3(residual)

        out  += residual
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################

class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, class_num=100):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = make_bn(16)
        self.relu = nn.ReLU(inplace=True)

        self.num_layers = layers

        self._make_group(block, 16, layers[0], group_id=1,
                         )
        self._make_group(block, 32, layers[1], group_id=2,
                         )
        self._make_group(block, 64, layers[2], group_id=3,
                         )

        # self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, make_bn):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()

    def _make_group(self, block, planes, layers, group_id=1
                    ):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer = self._make_layer_v2(block, planes, stride=stride,
                                       )

            # setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)
            # setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])


    def _make_layer_v2(self, block, planes, stride=1,
                       ):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, momentum=MOMENTUM,
                    quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)

        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        # if gate_type == 'ffgate1':
            # gate_layer = FeedforwardGateI(pool_size=pool_size,
                                          # channel=planes*block.expansion)
        # elif gate_type == 'ffgate2':
            # gate_layer = FeedforwardGateII(pool_size=pool_size,
                                           # channel=planes*block.expansion)
        # elif gate_type == 'softgate1':
            # gate_layer = SoftGateI(pool_size=pool_size,
                                   # channel=planes*block.expansion)
        # elif gate_type == 'softgate2':
            # gate_layer = SoftGateII(pool_size=pool_size,
                                    # channel=planes*block.expansion)
        # else:
            # gate_layer = None

        # if downsample:
            # return downsample, layer, gate_layer
        # else:
            # return None, layer, gate_layer

        return layer

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, momentum=MOMENTUM,
                    quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, num_bits=0):
        x = self.conv1(x, num_bits)
        x = self.bn1(x)
        x = self.relu(x)

        for g in range(3):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet20(class_num):
    # n = 6
    model = ResNet(BasicBlock, [3, 3, 3], class_num)
    return model

def resnet38(class_num):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], class_num)
    return model


def resnet74(class_num):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], class_num)
    return model


def resnet110(class_num):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], class_num)
    return model


def resnet164(class_num):
    # n = 27
    model = ResNet(BasicBlock, [27, 27, 27], class_num)
    return model