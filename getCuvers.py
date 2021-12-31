#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-12-23 16:19
@Desc   ：
==================================================
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import norm
import matplotlib.mlab as mlab
import cv2
import os
import shutil
import random
import glob
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model

# sn.set_style("white")
np.set_printoptions(precision=5)
import matplotlib
import math

fontsize = 16
styles = ['normal', 'italic', 'oblique']
weights = ['ultralight', 'light', 'normal', 'regular', 'book',
           'medium', 'roman', 'semibold', 'demibold', 'demi',
           'bold', 'heavy', 'extra bold', 'black']
font = {'family': 'Times New Roman',
        'style': styles[0],
        'weight': weights[10],
        'size': fontsize}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

CSV_DIR = "/home/zhucc/U-RCNN/runs/vis/2_OD_SS_e300_0.001_adamw_FeatureMap32"

ACC_ALL_Stomata = []
ACC_ALL_Cell = []
ACC_ALL_SI = []

plt.figure(figsize=(20, 12))
plt.title("Counting accuracy of stomata and cell", fontsize=fontsize)
for fold in range(5):
    ACC_FOLD_Stomata = []
    ACC_FOLD_Cell = []
    ACC_FOLD_SI = []
    for epoch in range(5, 301, 5):
        csv = os.path.join(CSV_DIR, f"fold-{fold}", f"epoch-{epoch}_si.csv")
        df = pd.read_csv(csv)
        ACC_FOLD_Stomata.append(np.array(df['stomata_accuracy(%)'])[-1])
        ACC_FOLD_Cell.append(np.array(df['cell_accuracy(%)'])[-1])
        ACC_FOLD_SI.append(np.array(df['si_accuracy(%)'])[-1])
    ACC_ALL_Stomata.append(ACC_FOLD_Stomata)
    ACC_ALL_Cell.append(ACC_FOLD_Cell)
    ACC_ALL_SI.append(ACC_FOLD_SI)

MEAN_Stomata = np.mean(np.array(ACC_ALL_Stomata), axis=0)
MEAN_Cell = np.mean(np.array(ACC_ALL_Cell), axis=0)
MEAN_SI = np.mean(np.array(ACC_ALL_SI), axis=0)
print(np.argmax(MEAN_Stomata), ": ", np.max(MEAN_Stomata))
print(np.argmax(MEAN_Cell), ": ", np.max(MEAN_Cell))
print(np.argmax(MEAN_SI), ": ", np.max(MEAN_SI))
ax1 = plt.subplot(311)
plt.plot(MEAN_Stomata, marker='o', label=f"Stomata", color=colors[0])
plt.tight_layout()
plt.legend()

ax2 = plt.subplot(312)
plt.plot(MEAN_Cell, marker='o', label=f"Cell", color=colors[1])
plt.tight_layout()
plt.legend()

ax3 = plt.subplot(313)
plt.plot(MEAN_SI, marker='o', label=f"SI", color=colors[2])
plt.tight_layout()
plt.legend()
plt.show()

# 2_OD_SS_e300_0.001_adamw_FeatureMap32
# 58*5 :  97.34478831330067
# 29*5 :  95.91647385395147
# 58*5 :  94.8511713364914

# 11_OD_SS_e100_0.001_adamw_FeatureMap32_wheat_resnet34
# 96 :  96.8823825686325
# 86 :  95.80583178400005
# 97 :  94.7895081165535

# 13_OD_SS_e100_0.001_adamw_FeatureMap32_wheat_bs_6_resnet34_768"
# 82 :  97.6834623357792
# 70 :  95.3901027643399
# 70 :  94.65713440359181