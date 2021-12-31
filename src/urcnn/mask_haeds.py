#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-11-24 11:17
@Desc   ：
==================================================
"""
import torch
from torch import nn


def mask_loss(mask_classes, masks_preds, targets):
    if mask_classes > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    target_masks = []
    for target in targets:
        target_masks.append(target["masks"])
    if len(target_masks) > 1:
        target_masks_tensor = torch.stack(target_masks, dim=0)
    else:
        target_masks_tensor = target_masks[0].unsqueeze(dim=0)
    mloss = criterion(masks_preds, target_masks_tensor.float())
    return mloss


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))


class MaskHeads(torch.nn.Module):
    def __init__(self, mask_classes, channel_num):
        self.mask_classes = mask_classes
        if self.mask_classes == 2:
            self.out_channel = 1
        else:
            self.out_channel = mask_classes
        super(MaskHeads, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # resnet50
        self.dconv_up3 = double_conv(channel_num[1] + channel_num[0], channel_num[1])
        self.dconv_up2 = double_conv(channel_num[1] + channel_num[2], channel_num[2])
        self.dconv_up1 = double_conv(channel_num[2] + channel_num[3], 64)
        # resnet34
        # self.dconv_up3 = double_conv(256 + 512, 256)
        # self.dconv_up2 = double_conv(128 + 256, 128)
        # self.dconv_up1 = double_conv(128 + 64, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, self.out_channel, 1),
        )

    def forward(self, features, pre_pools, masks):
        x = self.upsample(features)
        x = torch.cat([x, pre_pools['conv4']], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv3.shape)
        x = torch.cat([x, pre_pools['conv3']], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv2.shape)
        x = torch.cat([x, pre_pools['conv2']], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv1.shape)
        x = torch.cat([x, pre_pools['conv1']], dim=1)
        out = self.dconv_last(x)
        loss = {}
        if self.training:
            assert masks is not None
            mloss = mask_loss(self.mask_classes, out, masks)
            loss = {
                "loss_mask": mloss
            }
        return out, loss

    # def forward(self, masks_preds, targets=None):
    #     if self.mask_classes > 2:
    #         criterion = nn.CrossEntropyLoss()
    #     else:
    #         criterion = nn.BCEWithLogitsLoss()
    #
    #     mask_loss = {}
    #     result = []
    #     if self.training:
    #         target_masks = []
    #         for target in targets:
    #             target_masks.append(target["masks"])
    #         if len(target_masks) > 1:
    #             target_masks_tensor = torch.stack(target_masks, dim=0)
    #         else:
    #             target_masks_tensor = target_masks[0].unsqueeze(dim=0)
    #         mlosses = criterion(masks_preds, target_masks_tensor.float())
    #         mask_loss = {
    #             "loss_mask": mlosses
    #         }
    #     else:
    #         result = masks_preds.detach().sigmoid()
    #     return result, mask_loss
