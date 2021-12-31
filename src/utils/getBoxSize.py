#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-12-13 11:11
@Desc   ：
==================================================
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d = '/home/zhucc/U-RCNN/data/wheat/cache'
edge = []
min_area = 10000000
num_200 = 0
for dirs in ['/fold-0', '/fold-1', '/fold-2', '/fold-3', '/fold-4']:
    f = open(d + dirs + '/bbox_GT_height-768_width-768cache.pkl', 'rb')
    info = pickle.load(f)
    for i, _id in enumerate(info):
        id_eara = []
        for _object in info[_id]:
            area = (_object['bbox'][2] - _object['bbox'][0]) * (_object['bbox'][3] - _object['bbox'][1])
            if (area < 5):
                continue
            if area < min_area:
                min_area = area
            if area < 200:
                num_200 = num_200 + 1
            id_eara.append(area)
        edge.append(np.sqrt(np.mean(id_eara)))

plt.figure()
plt.hist(edge, bins=200, color=sns.desaturate("indianred", .8), alpha=.4)
plt.xlabel("area")
plt.ylabel("frequency")
plt.show()
print(min_area)
print(num_200)
