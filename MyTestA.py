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
import albumentations as A
import cv2


image = cv2.imread("/home/zhucc/11.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.Resize(width=image.shape[0]//2, height=image.shape[1]//2),
    A.HorizontalFlip(p=1),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

bboxes = [
    [23, 74, 295, 388],
    [377, 294, 512, 532],
    [333, 421, 632, 736],
]
class_labels = ['cat', 'dog', 'parrot']
transformed = transform(image=image, bboxes=bboxes, labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
print(transformed_bboxes)

