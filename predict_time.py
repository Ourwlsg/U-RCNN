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
import torchvision
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from src.urcnn.backbone.myResnet import resnet34, resnet50
from src.urcnn.urcnn import URCNN
from src.urcnn.rpn import AnchorGenerator
from src.utils.util import set_seed, K_FOLD, Meter
from skimage import segmentation, measure, color


def vis_detections(object_preds, image, threshold=0.5):
    """Visual debugging of detections."""
    num_stomata = 0
    scores = object_preds[0]['scores']
    bboxes = object_preds[0]['boxes']
    for i, score in enumerate(scores):
        if score > threshold:
            # bbox = bboxes[i].cpu().numpy()
            # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=2)
            # cv2.putText(img=image,
            #             text='%.3f' % score,
            #             org=(int(bbox[0]), int(bbox[1]) + 15),
            #             fontFace=cv2.FONT_HERSHEY_PLAIN,
            #             fontScale=0.5,
            #             color=(0, 0, 255),
            #             thickness=1,
            #             lineType=cv2.LINE_AA)
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

        # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.5)
        # rects.append(rect)
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

    model.load_state_dict(torch.load(args.load_name, map_location=device), strict=False)
    print(f'Model loaded from {args.load_name}')
    imagesDir = "/home/zhucc/U-RCNN/data/wheat/JPEGImages"
    imageList = os.listdir(imagesDir)
    num_images = len(imageList)
    start = time.time()
    print('Loaded Photo: {} images.'.format(num_images))
    infer_time = np.zeros((num_images, 1))
    model.eval()
    with open(os.path.join(args.vis_dir, "si_total.csv"), "w", encoding='utf-8-sig') as F:
        csv_write = csv.writer(F)
        csv_head = ['path', 'stoamta_DL', 'cell_DL', 'si_DL']
        csv_write.writerow(csv_head)
        for i, image_name in enumerate(tqdm(imageList)):
            if os.path.splitext(image_name)[1] not in ['.jpg', '.jpeg', '.png', '.tif', '.bmp']:
                print(image_name + "has been skipped")
                continue
            image_name = image_name.strip()
            im_file = os.path.join(imagesDir, image_name)
            image_BGR = cv2.imread(im_file, cv2.IMREAD_COLOR)
            image_512 = cv2.resize(image_BGR, (args.height, args.width), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.ToTensor()(image)
            img = img.unsqueeze(0)
            images_gpu = img.to(device)
            infer_start = time.time()
            with torch.no_grad():
                object_preds, mask_preds = model(images_gpu)
            infer_end = time.time()
            # ender.record()

            # WAIT FOR GPU SYNC
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            infer_time[i] = infer_end - infer_start
            vis_stomata_img, num_stomata = vis_detections(object_preds, image_512, threshold=args.vis_stomata_thre)
            vis_cell_img, count_coneara, num_cell, rects = vis_masks(mask_preds, threshold=args.vis_cell_thre)
            si = num_stomata / (num_stomata + num_cell) * 100
            line = [im_file, num_stomata, num_cell, si]
            csv_write.writerow(line)

    end = time.time()
    print(f"CPU Average counting time is {(end - start) / num_images} s!")
    mean_syn = np.sum(infer_time) / num_images
    std_syn = np.std(infer_time)
    print(f"CPU Average inference time is {mean_syn} , std is {std_syn}")

    # CPU
    # Average
    # counting
    # time is 0.5144233546414219
    # s!
    # CPU
    # Average
    # inference
    # time is 0.4478210805060266, std is 0.08372323968121555

    # GPU
    # Average
    # counting
    # time is 0.09153307734669505
    # s!
    # GPU
    # Average
    # inference
    # time is 0.02746813018600662, std is 0.003558193539795034