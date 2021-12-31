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


def vis_detections(object_preds, image, threshold=0.5):
    """Visual debugging of detections."""
    num_stomata = 0
    scores = object_preds[0]['scores']
    bboxes = object_preds[0]['boxes']
    for i, score in enumerate(scores):
        if score > threshold:
            bbox = bboxes[i].cpu().numpy()
            # cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #             1.5, (0, 0, 255), thickness=2)

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=2)
            cv2.putText(img=image,
                        text='%.3f' % score,
                        org=(int(bbox[0]), int(bbox[1]) + 15),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.0,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)
            num_stomata = num_stomata + 1
    return image, num_stomata


def count(img):
    cleared = img.copy()
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared, connectivity=2)  # 2代表8连通，1代表4联通区域标记
    props = measure.regionprops(label_image)
    count1 = len(props)
    count2 = len(props)

    # borders = np.logical_xor(img, cleared)  # 异或扩
    # label_image[borders] = -1
    dst = color.label2rgb(label_image, image=cleared, bg_label=0)  # 不同标记用不同颜色显示

    regions = measure.regionprops(label_image)
    areas = [region.area for region in regions]
    rects = []
    for region in regions:
        if region.area < (np.mean(areas) / 10):
            count2 -= 1
            continue
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.5)
        rects.append(rect)
    return dst, count1, count2, rects


def vis_masks(mask_preds, threshold):
    probs = torch.sigmoid(mask_preds).squeeze(0)
    mask = probs.squeeze().cpu().numpy() > threshold
    cell_pred = (mask * 255).astype(np.uint8)
    bifilter = cv2.bilateralFilter(cell_pred, 9, 75, 75)  # 双边滤波模糊去噪，保留边缘信息
    ret, binary = cv2.threshold(bifilter, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cells_count1, cells_count2, label_Unet, rects_Unet = count(opening.astype(np.uint8))
    return cells_count1, cells_count2, label_Unet, rects_Unet


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    with open("cfgs/predict.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # get model
    # backbone = resnet34(in_channel=3, pretrained=True)
    backbone = resnet34(in_channel=3, pretrained=True)
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

    for fold in args.FOLD:
        with open(os.path.join(args.data_dir, f"ImageSets/fold_{fold}", "val.txt")) as f:
            V_ids = f.readlines()
        num_images = len(V_ids)
        for epoch in range(1, 101):
            stomata_out_dir = os.path.join(args.vis_dir, args.experiment, f"fold-{fold}/epoch-{epoch}/stomata")
            cell_out_dir = os.path.join(args.vis_dir, args.experiment, f"fold-{fold}/epoch-{epoch}/cell")
            os.makedirs(stomata_out_dir, exist_ok=True)
            os.makedirs(cell_out_dir, exist_ok=True)

            load_file = os.path.join(args.load_dir, f"fold-{fold}", f"CP_epoch{epoch}.pth")
            model.load_state_dict(torch.load(load_file, map_location=device), strict=False)
            print(f'Model loaded from {load_file}')
            model.eval()
            with open(os.path.join(args.vis_dir, args.experiment, f"fold-{fold}/epoch-{epoch}_si.csv"), "w", encoding='utf-8-sig') as F:
                csv_write = csv.writer(F)
                csv_head = ['path', 'stoamta_DL', 'cell_DL', 'si_DL']
                if args.is_eval:
                    sum_stomata_GT = 0
                    sum_stomata_DL = 0
                    sum_stomata_error = 0
                    sum_stomata_accuracy = 0
                    csv_head.append('stomata_GT')
                    csv_head.append('stomata_error')
                    csv_head.append('stomata_accuracy(%)')

                    sum_cell_GT = 0
                    sum_cell_DL = 0
                    sum_cell_error = 0
                    sum_cell_accuracy = 0
                    csv_head.append('cell_GT')
                    csv_head.append('cell_error')
                    csv_head.append('cell_accuracy(%)')

                    sum_si_GT = 0
                    sum_si_DL = 0
                    sum_si_error = 0
                    sum_si_accuracy = 0
                    csv_head.append('si_GT')
                    csv_head.append('si_error')
                    csv_head.append('si_accuracy(%)')
                csv_write.writerow(csv_head)
                for image_name in tqdm(V_ids, desc=f"validating fold-{fold}-epoch{epoch}: "):
                    image_name = image_name.strip()
                    im_file = os.path.join(args.data_dir, f"JPEGImages/{image_name}.jpg")
                    image_BGR = cv2.imread(im_file, cv2.IMREAD_COLOR)
                    image_512 = cv2.resize(image_BGR, (args.height, args.width), interpolation=cv2.INTER_LINEAR)
                    image = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)
                    img = torchvision.transforms.ToTensor()(image)
                    img = img.unsqueeze(0)
                    images_gpu = img.to(device)
                    with torch.no_grad():
                        object_preds, mask_preds = model(images_gpu)
                    vis_stomata_img, num_stomata = vis_detections(object_preds, image_512, threshold=args.vis_stomata_thre)
                    vis_cell_img, count_coneara, num_cell, rects = vis_masks(mask_preds, threshold=args.vis_cell_thre)
                    si_DL = num_stomata / (num_stomata + num_cell) * 100
                    line = [im_file, num_stomata, num_cell, si_DL]
                    sum_stomata_DL = sum_stomata_DL + num_stomata
                    sum_cell_DL = sum_cell_DL + num_cell
                    sum_si_DL = sum_si_DL + si_DL
                    if args.is_SaveVis:
                        cv2.imwrite(os.path.join(stomata_out_dir, f"{image_name}_stomata_out.jpg"), vis_stomata_img)
                        cv2.imwrite(os.path.join(cell_out_dir, f"{image_name}_cell_out.jpg"), cv2.cvtColor((vis_cell_img* 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if args.is_eval:
                        # 获得实际气孔个数
                        anno_xml = os.path.join(args.data_dir, "Annotations", f"{image_name}.xml")
                        tree = ET.parse(anno_xml)
                        root = tree.getroot()
                        stomata_count_GT = len(list(root.iter('object')))
                        sum_stomata_GT = sum_stomata_GT + stomata_count_GT
                        stomata_error = abs(stomata_count_GT - num_stomata)
                        stomata_accuracy = (1 - stomata_error / stomata_count_GT) * 100
                        line.append(stomata_count_GT)
                        line.append(stomata_error)
                        sum_stomata_error = sum_stomata_error + stomata_error
                        line.append(stomata_accuracy)
                        sum_stomata_accuracy = sum_stomata_accuracy + stomata_accuracy

                        # 获得实际细胞个数
                        mask_file = os.path.join(args.data_dir, "Masks", f"{image_name}.jpg")
                        mask_true = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        ret, mask_true = cv2.threshold(mask_true, 230, 255, cv2.THRESH_BINARY)
                        vis_cell_img_GT, count_coneara_GT, num_cell_GT, rects_GT = count(mask_true)
                        sum_cell_GT = sum_cell_GT + num_cell_GT
                        cell_error = abs(num_cell_GT - num_cell)
                        cell_accuracy = (1 - cell_error / num_cell_GT) * 100
                        line.append(num_cell_GT)
                        line.append(cell_error)
                        sum_cell_error = sum_cell_error + cell_error
                        line.append(cell_accuracy)
                        sum_cell_accuracy = sum_cell_accuracy + cell_accuracy

                        # 获得实际气孔指数
                        si_GT = stomata_count_GT / (num_cell_GT + stomata_count_GT) * 100
                        line.append(si_GT)
                        sum_si_GT = sum_si_GT + si_GT
                        si_error = abs(si_GT - si_DL)
                        line.append(si_error)
                        sum_si_error = + sum_si_error + si_error
                        si_accuracy = (1 - si_error / si_GT) * 100
                        line.append(si_accuracy)
                        sum_si_accuracy = sum_si_accuracy + si_accuracy
                    csv_write.writerow(line)
                if args.is_eval:
                    average_line = ['average',
                                    sum_stomata_DL/num_images,
                                    sum_cell_DL / num_images,
                                    sum_si_DL / num_images,

                                    sum_stomata_GT / num_images,
                                    sum_stomata_error / num_images,
                                    sum_stomata_accuracy / num_images,

                                    sum_cell_GT / num_images,
                                    sum_cell_error / num_images,
                                    sum_cell_accuracy / num_images,

                                    sum_si_GT / num_images,
                                    sum_si_error / num_images,
                                    sum_si_accuracy / num_images
                                    ]
                    csv_write.writerow(average_line)