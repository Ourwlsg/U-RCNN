import bisect
import glob
import os
import re
import time
import torch
import torchvision
import yaml
import cv2
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import misc
import numpy as np
from U_RCNN_pytorch import Meter
import U_RCNN_pytorch as urp
from U_RCNN_pytorch.datasets.voc_dataset import BatchCollator


def K_FOLD(PATH):
    with open(PATH + '/train.txt') as f1:
        T_ids = f1.readlines()
    with open(PATH + '/val.txt') as f2:
        V_ids = f2.readlines()
    return T_ids, V_ids


if __name__ == "__main__":
    with open("./cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # for fold in range(args.K_FOLD):
    for fold in range(1):
        train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{fold}'))
        dataset_train = urp.datasets("voc", data_dir=args.data_dir, ids=train_ids, train=True)

        train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=0,
                                  pin_memory=True, collate_fn=BatchCollator(train=dataset_train.train))

        # model = urp.URCNN_resnet50(pretrained=True, num_classes=2).to(device)

        # faster rcnn
        # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # backbone.out_channels = 1280
        # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        # model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).to(device)

        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        model = MaskRCNN(backbone,
                         num_classes=2,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         mask_roi_pool=mask_roi_pooler).to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_lambda = lambda x: 0.1 ** bisect.bisect([22, 26], x)
        start_epoch = 0
        for epoch in range(start_epoch, args.epochs):
            iters = len(train_loader) if args.iters < 0 else args.iters

            t_m = Meter("total")
            m_m = Meter("model")
            b_m = Meter("backward")
            model.train()
            A = time.time()
            for i, (images, targets) in enumerate(train_loader):
                T = time.time()
                num_iters = epoch * len(train_loader) + i
                images = images.to(device)
                # Optional[List[Dict[str, Tensor]]])
                targets_gpu = []
                for target in targets:
                    target_gpu = {}
                    for k, v in target.items():
                        if k == "labels":
                            target_gpu[k] = v.to(device, torch.int64)
                        else:
                            target_gpu[k] = v.to(device)
                    # target_gpu = {k: v.to(device).int64() if k=="lables" else k: v.to(device) for k, v in target.items()}
                    targets_gpu.append(target_gpu)

                S = time.time()

                losses = model(images, targets_gpu)
                total_loss = sum(losses.values())
                m_m.update(time.time() - S)

                S = time.time()
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                b_m.update(time.time() - S)

                if num_iters % args.print_freq == 0:
                    print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

                t_m.update(time.time() - T)
                if i >= iters - 1:
                    break

            A = time.time() - A
            print(
                "iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg,
                                                                                      1000 * m_m.avg, 1000 * b_m.avg))