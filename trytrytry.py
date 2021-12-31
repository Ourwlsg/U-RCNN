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
# import src.models.resnet50 as resnet
# import torch
# import torch.nn as nn
# import numpy as np
# if __name__ == "__main__":
#     model = resnet.resnet50(pretrained=True)
#     # for name, param in model.named_parameters():
#     #     print(name, param.requires_grad)
#     model.eval()
#     x = torch.rand(2, 3, 300, 400)
#     for index, child in enumerate(model.children()):
#         print(index, child)
#     print(nn.Sequential(*list(model.children()))[:3])
#     predictions = model(x)
#     print(predictions)



# import torch
# from torchvision.models.resnet import resnet101
# net = resnet101(pretrained=False)
# input = torch.rand([14, 3, 512, 512])
# output = net(input)
# from src import get_gpu_prop
#
# get_gpu_prop(show=True)
import math

import cv2
from pylab import *

Image = cv2.imread('02.jpg', 1)  # 读入原图
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
img = np.array(image, dtype=np.float64)  # 读入到np的array中，并转化浮点类型

# 初始水平集函数
IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
IniLSF[300:320, 300:320] = -1
IniLSF = -IniLSF

# 画初始轮廓
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
plt.figure(1), plt.imshow(Image), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.contour(IniLSF, [0], color='b', linewidth=2)  # 画LSF=0处的等高线
plt.draw(), plt.show(block=False)


def mat_math(intput, str):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output


# CV函数
def CV(LSF, img, mu, nu, epison, step):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * mat_math(LSF / epison, "atan"))
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix * Ix + Iy * Iy, "sqrt")
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    LSF = LSF + step * (Length + Penalty + CVterm)
    # plt.imshow(s, cmap ='gray'),plt.show()
    return LSF


# 模型参数
mu = 1
nu = 0.003 * 255 * 255
num = 20
epison = 1
step = 0.1
LSF = IniLSF
for i in range(1, num):
    LSF = CV(LSF, img, mu, nu, epison, step)  # 迭代
    if i % 1 == 0:  # 显示分割轮廓
        plt.imshow(Image), plt.xticks([]), plt.yticks([])
        plt.contour(LSF, [0], colors='r', linewidth=2)
        plt.draw(), plt.show(block=False), plt.pause(0.01)