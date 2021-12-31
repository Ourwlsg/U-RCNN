#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-11-24 10:32
@Desc   ：
==================================================
"""
import csv
import os
import time

import cv2
import yaml
import torch
import logging
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import datasets
from src.urcnn.backbone.myResnet import resnet34, resnet50
from src.urcnn.loss.muti_loss import my_MultiTaskLossWrapper
from src.urcnn.urcnn import URCNN
from src.urcnn.rpn import AnchorGenerator
from src.datasets.voc_dataset import BatchCollator
import matplotlib.patches as mpatches
from albumentations import Compose, Resize, BboxParams, RandomRotate90, Flip, ShiftScaleRotate, Rotate
from src.utils.gpu import toDevice, get_gpu_prop
from src.utils.util import set_seed, K_FOLD, Meter
from test import validation
import xml.etree.ElementTree as ET
from skimage import segmentation, measure, color
from torchsummary import summary

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    if args.backbone == "resnet34":
        backbone = resnet34(in_channel=3, pretrained=True)
    elif args.backbone == "resnet50":
        backbone = resnet50(in_channel=3, pretrained=True)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=(args.anchorsizes,), aspect_ratios=(args.aspect_ratios,))
    loss_handler = None
    mask_classes = 2
    model = URCNN(backbone,
                  num_classes=args.num_classes,
                  min_size=args.height,
                  max_size=args.width,
                  image_mean=args.norm_mean,
                  image_std=args.norm_std,
                  rpn_pre_nms_top_n_train=args.rpn_pre_nms_top_n_train,
                  rpn_pre_nms_top_n_test=args.rpn_pre_nms_top_n_test,
                  rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
                  rpn_post_nms_top_n_test=args.rpn_post_nms_top_n_test,
                  box_score_thresh=args.train_score_thre,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_classes=mask_classes,
                  loss_handler=loss_handler).to(device)
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # for name, child in model.named_children():
    #     print(name)
    #     total_params = sum(p.numel() for p in child.parameters())
    #     print(f'{total_params:,} total parameters.')

    train_Transform = Compose([Resize(height=args.height,
                                      width=args.width,
                                      interpolation=Image.LINEAR, p=1),
                               RandomRotate90(p=0.2),
                               Rotate(limit=(45, 45), p=0.2),
                               Flip(p=0.5),
                               # ShiftScaleRotate(p=0.2, interpolation=Image.NEAREST),
                               ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"], min_area=200))

    train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{0}'))
    dataset_train = datasets(args.dataset,
                             train=True,
                             ids=train_ids,
                             data_dir=args.data_dir,
                             transform=train_Transform)

    train_loader = DataLoader(shuffle=True,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True,
                              dataset=dataset_train,
                              batch_size=1,
                              collate_fn=BatchCollator(train=dataset_train.train))

    for i, (images, targets) in enumerate(train_loader):
        image = images.squeeze().permute(1, 2, 0).numpy()
        boxes = targets[0]["boxes"].numpy()
        mask = (targets[0]["masks"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        for i, bbox in enumerate(boxes):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          color=(255, 0, 0), thickness=2)
            print((int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)

        plt.subplot(122)
        plt.imshow(mask)
        plt.show()
        plt.close()

















# resnet50                     # resnet34
# total:       133,175,800       38,963,704
# transform:   0                 0
# backbone:    23,508,032        21,284,672
# rpn:         9,484,333         601,645
# roi_heads:   52,441,098        13,905,930
# mask_heads:  47,742,337        3,171,457
