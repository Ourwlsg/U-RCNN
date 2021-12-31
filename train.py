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
import os
import time
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

from src import datasets
from src.urcnn.backbone.myResnet import resnet34, resnet50
from src.urcnn.loss.muti_loss import my_MultiTaskLossWrapper
from src.urcnn.urcnn import URCNN
from src.urcnn.rpn import AnchorGenerator
from src.datasets.voc_dataset import BatchCollator

from albumentations import Compose, Resize, BboxParams, RandomRotate90, Flip, ShiftScaleRotate, Rotate

# from albumentations import (
#     HorizontalFlip, VerticalFlip, IAAPerspective, CLAHE, RandomRotate90, ElasticTransform, RandomGamma,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomSizedCrop, PadIfNeeded,
#     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ToGray, Resize, Normalize, BboxParams
# )
from src.utils.gpu import toDevice, get_gpu_prop
from src.utils.util import set_seed, K_FOLD, Meter
from test import validation

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # log setup
    SS = args.SS
    OD = args.OD

    logdir = os.path.join(args.log_dir, args.experiment)
    os.makedirs(logdir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_filename = f"{logdir}/{timestamp}_" \
                   f"LR-{args.lr}_" \
                   f"OPTIM-{args.optim}_" \
                   f"HEIGHT-{args.height}_" \
                   f"WIDTH-{args.width}_" \
                   f"BS-{args.batch_size}_" \
                   f"EPOCHS-{args.epochs}_LOG.txt"
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
                        filename=log_filename)

    logging.info("========================cfgs==========================")
    print("========================cfgs==========================")
    for item in args:
        logging.info("{0:20} ====> {1}".format(item, args[item]))
        print("{0:20} ====> {1}".format(item, args[item]))
    print("========================cfgs==========================\n")
    logging.info("========================cfgs==========================\n")

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    device_ids = range(torch.cuda.device_count())
    logging.info(get_gpu_prop(show=True))
    logging.info(f'Using device {device} {args.gpus}')

    # dataset setup
    train_Transform = Compose([Resize(height=args.height,
                                      width=args.width,
                                      interpolation=Image.LINEAR, p=1),
                               RandomRotate90(p=0.2),
                               Rotate(limit=(45, 45), p=0.2),
                               Flip(p=0.5),
                               # ShiftScaleRotate(p=0.2, interpolation=Image.NEAREST),
                               ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"], min_area=200))

    valid_Transform = Compose([Resize(height=args.height,
                                      width=args.width,
                                      interpolation=Image.NEAREST, p=1),
                               ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    for fold in args.FOLD:
        # for fold in range(1, 2):
        logging.info(f"Start fold-{fold}!")
        print(f"Start fold-{fold}!")
        writer = SummaryWriter(log_dir=args.board_dir + f"/{args.experiment}/fold-{fold}",
                               comment=f'FOLD-{fold}/LR-{args.lr}_'
                                       f'BS-{args.batch_size}_'
                                       f'EPOCHS-{args.epochs}_'
                                       f'HEIGHT-{args.height}_'
                                       f'WIDTH-{args.width}')

        train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{fold}'))
        logging.info(f"{len(train_ids)} images for training")
        logging.info(f"{len(val_ids)} images for validation")

        dataset_train = datasets(args.dataset,
                                 train=True,
                                 ids=train_ids,
                                 data_dir=args.data_dir,
                                 transform=train_Transform)

        dataset_valid = datasets(args.dataset,
                                 train=False,
                                 ids=val_ids,
                                 data_dir=args.data_dir,
                                 transform=valid_Transform)

        train_loader = DataLoader(shuffle=True,
                                  num_workers=0,
                                  drop_last=True,
                                  pin_memory=True,
                                  dataset=dataset_train,
                                  batch_size=args.batch_size,
                                  collate_fn=BatchCollator(train=dataset_train.train))

        valid_loader = DataLoader(shuffle=False,
                                  num_workers=0,
                                  drop_last=False,
                                  pin_memory=True,
                                  dataset=dataset_valid,
                                  batch_size=1,  # only support 1
                                  collate_fn=BatchCollator(train=dataset_valid.train))

        # get model
        if args.backbone == "resnet34":
            backbone = resnet34(in_channel=3, pretrained=True)
        elif args.backbone == "resnet50":
            backbone = resnet50(in_channel=3, pretrained=True)
        if OD:
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
            anchor_generator = AnchorGenerator(sizes=(args.anchorsizes,), aspect_ratios=(args.aspect_ratios,))
        else:
            roi_pooler = None
            anchor_generator = None
        if args.use_lossWeigth:
            loss_handler = my_MultiTaskLossWrapper(args.weight_num)
        else:
            loss_handler = None

        if SS:
            mask_classes = 2
        else:
            mask_classes = None

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

        # graph_inputs = torch.rand(14, 3, args.inputsize, args.inputsize).to(device)
        # writer.add_graph(model, input_to_model=graph_inputs, verbose=True)
        for index, child in enumerate(model.children()):
            print(index, child)
        if args.load:
            model.load_state_dict(torch.load(args.load_name, map_location=device), strict=False)
            logging.info(f'Model loaded from {args.load_name}')
            print(f'Model loaded from {args.load_name}')

        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

        # 不同部分的参数设置不同的学习率（以及其他属性）
        # params_dict = [{'params': model.layer.parameters(), 'lr': 0.1},
        #                {'params': model.layer2.parameters(), 'lr': 0.2}]
        # optimizer_Adam = torch.optim.Adam(params_dict)


        # 只传入layer层的参数，就可以只更新layer层的参数而不影响其他参数。
        # optimizer_Adam = torch.optim.Adam(model.layer.parameters(), lr=0.1)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = {
            "sgd": lambda: torch.optim.SGD(params, lr=args.lr, momentum=0.9,),
            "adam": lambda: torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-08),
            "adamw": lambda: torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4),
            "rmsprop": lambda: torch.optim.RMSprop(params, lr=args.lr, momentum=0.9, eps=0.001, weight_decay=1e-8)
        }[args.optim]()

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs * len(dataloader), eta_min=0, last_epoch=-1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)
        scheduler = LambdaLR(optimizer, lambda x: (((1 + np.cos(x * np.pi / args.epochs)) / 2) ** 1.0) * 0.9 + 0.1)

        for epoch in range(args.start_epoch, args.epochs + 1):
            print("Start validating...")
            ap, dice = validation(fold, epoch - 1,
                                  val_ids, valid_loader,
                                  model, device,
                                  args, writer)

            iters = len(train_loader) if args.iters < 0 else args.iters
            t_m = Meter("total")
            m_m = Meter("model")
            b_m = Meter("backward")
            model.train()
            logging.info(f"start epoch_{epoch}...")
            print(f"start epoch_{epoch}...")
            if not (SS or OD):
                raise RuntimeError("OD and SS cannot be false at the same time")
            if SS and OD:
                print('iter  ', 'classifier  ', 'box_reg  ', 'objectness  ', 'rpn_box_reg  ', 'mask  ', 'total')
            elif OD:
                print('iter  ', 'classifier  ', 'box_reg  ', 'objectness  ', 'rpn_box_reg  ', 'total')
            else:
                print('iter  ', 'mask_loss', )

            A = time.time()
            for i, (images, targets) in enumerate(train_loader):
                T = time.time()
                num_iters = (epoch - 1) * len(train_loader) + i + 1
                images_gpu = images.to(device)
                targets_gpu = toDevice(targets, device, OD, SS)

                S = time.time()
                losses, weighted_loss, weights = model(images_gpu, targets_gpu, OD, SS)
                if args.use_lossWeigth and SS and OD:
                    total_loss = weighted_loss
                else:
                    total_loss = sum(losses.values())

                m_m.update(time.time() - S)

                S = time.time()
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                b_m.update(time.time() - S)
                if num_iters % args.print_freq == 0:
                    if args.use_lossWeigth:
                        print("weights:", '\t'.join([str(w) for w in weights]))
                        if args.use_lossWeigth and args.weight_num == 2:
                            writer.add_scalar("weights/frcnn", weights[0], num_iters)
                            writer.add_scalar("weights/mask", weights[1], num_iters)
                        if args.use_lossWeigth and args.weight_num == 3:
                            writer.add_scalar("weights/rpn", weights[0], num_iters)
                            writer.add_scalar("weights/rcnn", weights[2], num_iters)
                            writer.add_scalar("weights/mask", weights[3], num_iters)
                        if args.use_lossWeigth and args.weight_num == 5:
                            writer.add_scalar("weights/rpn_objectness", weights[0], num_iters)
                            writer.add_scalar("weights/rpn_box_reg", weights[1], num_iters)
                            writer.add_scalar("weights/rcnn_classifier", weights[2], num_iters)
                            writer.add_scalar("weights/rcnn_box_reg", weights[3], num_iters)
                            writer.add_scalar("weights/mask", weights[4], num_iters)

                    if SS and OD:
                        print(f'{num_iters}\t',
                              '{:.4f}\t'.format(losses['loss_classifier'].item()),
                              '{:.4f}\t'.format(losses['loss_box_reg'].item()),
                              '{:.4f}\t'.format(losses['loss_objectness'].item()),
                              '{:.4f}\t'.format(losses['loss_rpn_box_reg'].item()),
                              '{:.4f}\t'.format(losses['loss_mask'].item()),
                              '{:.4f}'.format(total_loss))
                    elif OD:
                        print(f'{num_iters}\t',
                              '{:.4f}\t'.format(losses['loss_classifier'].item()),
                              '{:.4f}\t'.format(losses['loss_box_reg'].item()),
                              '{:.4f}\t'.format(losses['loss_objectness'].item()),
                              '{:.4f}\t'.format(losses['loss_rpn_box_reg'].item()),
                              '{:.4f}'.format(total_loss))
                    else:
                        print(f'{num_iters}\t\t',
                              '{:.4f}'.format(losses['loss_mask'].item()))
                    # for tag, value in model.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), num_iters)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), num_iters)
                    info = {'loss': total_loss}
                    info.update(losses)
                    writer.add_scalars("losses/train", info, num_iters)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], num_iters)

                t_m.update(time.time() - T)
                if i >= iters - 1:
                    break

            scheduler.step()

            if epoch % args.save_freq == 0:
                os.makedirs(args.checkpoint_dir + f"/{args.experiment}/fold-{fold}/", exist_ok=True)
                if len(device_ids) > 1:
                    torch.save(model.module.state_dict(),
                               args.checkpoint_dir + f"/{args.experiment}/fold-{fold}/" + f'CP_epoch{epoch}.pth')
                else:
                    torch.save(model.state_dict(),
                               args.checkpoint_dir + f"/{args.experiment}/fold-{fold}/" + f'CP_epoch{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')
            A = time.time() - A
            print("epoch: {:.1f}ms\t total: {:.1f}ms\t model: {:.1f}ms\t backward: {:.1f}ms".
                  format(1000 * A,
                         1000 * t_m.avg,
                         1000 * m_m.avg,
                         1000 * b_m.avg))
        # final epoch validation
        ap, dice = validation(fold, args.epochs, val_ids, valid_loader, model, device, args, writer)
