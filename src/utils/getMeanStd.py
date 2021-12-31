#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-12-13 10:39
@Desc   ：
==================================================
"""

import numpy as np
import cv2
import os

img_h, img_w = 768, 768  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = '/home/zhucc/U-RCNN/data/wheat/JPEGImages'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list) - 1
i = 0
for item in imgs_path_list:
    if item.endswith('.keep'):
        continue
    img = cv2.imread(os.path.join(imgs_path, item))
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))