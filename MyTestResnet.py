#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：kaggle -> 12
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2020/5/13 21:17
@Desc   ：
==================================================
"""
import U_RCNN_pytorch.models.resnet50 as resnet
import torch
import torch.nn as nn
import numpy as np
if __name__ == "__main__":
    model = resnet.resnet50(pretrained=True)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    model.eval()
    x = torch.rand(2, 3, 300, 400)
    for index, child in enumerate(model.children()):
        print(index, child)
    print(nn.Sequential(*list(model.children()))[:3])
    predictions = model(x)
    print(predictions)
