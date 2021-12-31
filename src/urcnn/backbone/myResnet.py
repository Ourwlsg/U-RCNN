#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-11-23 11:34
@Desc   ：
==================================================
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#     """
#
#     # BasicBlock and BottleNeck block
#     # have different output size
#     # we use class attribute expansion
#     # to distinct
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         # residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )
#
#         # shortcut
#         self.shortcut = nn.Sequential()
#
#         # the shortcut output dimension is not the same with residual function
#         # use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
#
#
# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#     """
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )
#
#         self.shortcut = nn.Sequential()
#
#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, in_channel, block, layers, channel_num, out_channels,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channel = in_channel
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(self.in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        # self.layer1 = self._make_layer(block, 64, layers[0], 1)
        # self.layer2 = self._make_layer(block, 128, layers[1], 2)
        # self.layer3 = self._make_layer(block, 256, layers[2], 2)
        # self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # feature = bottle
        # self.out_channels = 512

        # feature = conv4 and resnet34
        # self.out_channels = 256
        self.channel_num = channel_num

        # feature = conv4 and resnet50
        self.out_channels = out_channels

    # def _make_layer(self, block, out_channels, num_blocks, stride):
    #     """make resnet layers(by layer i didnt mean this 'layer' was the
    #     same as a neuron netowork layer, ex. conv layer), one layer may
    #     contain more than one residual block
    #     Args:
    #         block: block type, basic block or bottle neck block
    #         out_channels: output depth channel number of this layer
    #         num_blocks: how many blocks per layer
    #         stride: the stride of the first block of this layer
    #
    #     Return:
    #         return a resnet layer
    #     """
    #
    #     # we have num_block blocks per layer, the first block
    #     # could be 1 or 2, other blocks would always be 1
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels * block.expansion
    #
    #     return nn.Sequential(*layers)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, with_prepools=False):
        pre_pools = dict()

        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        conv1 = self.relu(x2)
        temp = self.maxpool(conv1)
        conv2 = self.layer1(temp)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        if with_prepools:
            pre_pools[f"conv1"] = conv1
            pre_pools[f"conv2"] = conv2
            pre_pools[f"conv3"] = conv3
            pre_pools[f"conv4"] = conv4
        bottle = self.layer4(conv4)

        # x = self.avgpool(bottle)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return conv4, bottle, pre_pools

    # def load_pretrained_weights(self):
    #     model_dict = models.resnet34(True).state_dict()[-2]
    # resnet34_weights = models.resnet34(True).state_dict()
    # count_res = 0
    # count_my = 0
    #
    # reskeys = list(resnet34_weights.keys())
    # mykeys = list(model_dict.keys())
    # # print(self)
    # # print(models.resnet34())
    # print(reskeys)
    # print("============")
    # print(mykeys)
    #
    # corresp_map = []
    # while (True):
    #     reskey = reskeys[count_res]
    #     if "fc" in reskey:
    #         break
    #     mykey = mykeys[count_my]
    #     # while reskey.split(".")[-1] not in mykey:
    #     #     count_my += 1
    #     #     mykey = mykeys[count_my]
    #
    #     corresp_map.append([reskey, mykey])
    #     count_res += 1
    #     count_my += 1
    # for k_res, k_my in corresp_map:
    #     model_dict[k_my] = resnet34_weights[k_res]

    # try:
    #     self.load_state_dict(model_dict)
    #     print("Loaded resnet34 weights in mynet !")
    # except:
    #     print("Error resnet34 weights in mynet !")
    #     raise


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _resnet(in_channel, arch, block, layers, channel_num, out_channels, pretrained, progress, **kwargs):
    model = ResNet(in_channel, block,
                   layers, channel_num, out_channels,
                   replace_stride_with_dilation=[False, False, False], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(in_channel, pretrained=True, progress=True, **kwargs):
    """ return a ResNet 18 object
    """
    return _resnet(in_channel, 'resnet18', BasicBlock, [2, 2, 2, 2],
                   pretrained, progress, **kwargs)


def resnet34(in_channel, pretrained=True, progress=True, **kwargs):
    r"""ResNet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
    return _resnet(in_channel, 'resnet34', BasicBlock, [3, 4, 6, 3], [512, 256, 128, 64], 256,
                   pretrained, progress, **kwargs)


def resnet50(in_channel, pretrained=True, progress=True, **kwargs):
    """ return a ResNet 50 object
    """
    return _resnet(in_channel, 'resnet50', Bottleneck, [3, 4, 6, 3], [2048, 1024, 512, 256], 1024,
                   pretrained, progress, **kwargs)

def resnet101(in_channel, pretrained=True, progress=True, **kwargs):
    """ return a ResNet 101 object
    """
    return _resnet(in_channel, 'resnet101', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def resnet152(in_channel, pretrained=True, progress=True, **kwargs):
    """ return a ResNet 152 object
    """
    return _resnet(in_channel, 'resnet152', Bottleneck, [3, 8, 36, 3],
                   pretrained, progress, **kwargs)


def resnext50_32x4d(in_channel, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(in_channel, 'resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(in_channel, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(in_channel, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(in_channel, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(in_channel, 'wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(in_channel, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(in_channel, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = resnet34(3, pretrained=True)
    print(net)
    x = torch.rand((1, 3, 512, 512))
    print(net(x, with_prepools=False).shape)
