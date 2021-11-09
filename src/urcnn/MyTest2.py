import torch
import torch.nn as nn
import torchvision
from torchvision.ops import misc

from src.urcnn.backbone import UNetWithResnet50Encoder
from src.urcnn.rpn import AnchorGenerator
from src.urcnn.urcnn import URCNN


class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = torchvision.models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x


# if __name__ == "__main__":
#     # load a pre-trained model for classification and return
#     # only the features
#     # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#     # # MaskRCNN needs to know the number of
#     # # output channels in a backbone. For mobilenet_v2, it's 1280
#     # # so we need to add it here
#     # backbone.out_channels = 1280
#
#     # backbone = ResBackbone('resnet50', pretrained = True)
#     backbone = UNetWithResnet50Encoder()
#
#     # let's make the RPN generate 5 x 3 anchors per spatial
#     # location, with 5 different sizes and 3 different aspect
#     # ratios. We have a Tuple[Tuple[int]] because each feature
#     # map could potentially have different sizes and
#     # aspect ratios
#     anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                        aspect_ratios=((0.5, 1.0, 2.0),))
#
#     # let's define what are the feature maps that we will
#     # use to perform the region of interest cropping, as well as
#     # the size of the crop after rescaling.
#     # if your backbone returns a Tensor, featmap_names is expected to
#     # be ['0']. More generally, the backbone should return an
#     # OrderedDict[Tensor], and in featmap_names you can choose which
#     # feature maps to use.
#     roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
#                                                     output_size=7,
#                                                     sampling_ratio=2)
#
#     mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
#                                                          output_size=14,
#                                                          sampling_ratio=2)
#     # put the pieces together inside a MaskRCNN model
#     model = URCNN(backbone,
#                      num_classes=2,
#                      rpn_anchor_generator=anchor_generator,
#                      box_roi_pool=roi_pooler)
#
#     model.eval()
#     # x = [torch.rand(3, 300, 400), torch.rand(3, 300, 400)]
#     x = torch.rand((2, 3, 512, 512))
#     for index, child in enumerate(model.children()):
#         print(index, child)
#     predictions, masks_preds = model(x)
#     print(len(predictions))


import bisect
import glob
import os
import re
import time
import torch
import torchvision
import yaml
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchvision.ops import misc
from PIL import Image
from src import Meter
import src as urp
from src.datasets.voc_dataset import BatchCollator
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, CLAHE, RandomRotate90, ElasticTransform, RandomGamma,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomSizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ToGray, Resize, Normalize, BboxParams
)


def K_FOLD(PATH):
    with open(PATH + '/train.txt') as f1:
        T_ids = f1.readlines()
    with open(PATH + '/val.txt') as f2:
        V_ids = f2.readlines()
    return T_ids, V_ids


if __name__ == "__main__":
    with open("../../cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    device_ids = range(torch.cuda.device_count())

    # Albumentations
    train_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        # RandomRotate90(0.5),
        # Flip(p=0.5),
        # ShiftScaleRotate(p=0.2, interpolation=Image.NEAREST),  # , border_mode=cv2.BORDER_CONSTANT, value=0
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    valid_Transform = Compose([
        # ToGray(p=1),
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        # Normalize(mean=norm_mean, std=norm_std),
        # # ToTensorV2()
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    # for fold in range(args.K_FOLD):
    for fold in range(1):
        train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{fold}'))
        dataset_train = urp.datasets("voc", data_dir=args.data_dir, ids=train_ids, transform=train_Transform,
                                     train=True)

        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True, collate_fn=BatchCollator(train=dataset_train.train), drop_last=True)

        # model = urp.URCNN_resnet50(pretrained=True, num_classes=2).to(device)

        # faster rcnn
        # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # backbone.out_channels = 1280
        # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        # model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).to(device)

        # #########
        # 加载URCNN
        # #########
        backbone = UNetWithResnet50Encoder()
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = URCNN(backbone,
                      num_classes=2,
                      min_size=args.inputsize,
                      max_size=args.inputsize,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler).to(device)
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

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
                        if isinstance(v, np.ndarray):
                            v = torch.from_numpy(v)
                        if k == "labels":
                            target_gpu[k] = v.to(device, torch.int64)
                        else:
                            target_gpu[k] = v.to(device)
                    # target_gpu = {k: v.to(device).int64() if k=="lables" else k: v.to(device) for k, v in target.items()}
                    targets_gpu.append(target_gpu)

                S = time.time()

                losses, mask_preds = model(images, targets_gpu)
                if backbone.n_classes > 1:
                    criterion = nn.CrossEntropyLoss()
                else:
                    criterion = nn.BCEWithLogitsLoss()
                mask_loss = {}
                if model.training:
                    target_masks = []
                    for target in targets:
                        target_masks.append(target["masks"])
                    if len(target_masks) > 1:
                        target_masks_tensor = torch.stack(target_masks, dim=0)
                    else:
                        target_masks_tensor = target_masks[0]
                    mask_losses = criterion(mask_preds, target_masks_tensor.to(device))
                    mask_loss = {
                        "loss_mask": mask_losses
                    }

                losses.update(mask_loss)
                total_loss = sum(losses.values())
                m_m.update(time.time() - S)

                S = time.time()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                b_m.update(time.time() - S)

                if num_iters % args.print_freq == 0:
                    print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()), "{}\t".format(total_loss))

                t_m.update(time.time() - T)
                if i >= iters - 1:
                    break

            A = time.time() - A
            print(
                "iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg,
                                                                                      1000 * m_m.avg, 1000 * b_m.avg))
