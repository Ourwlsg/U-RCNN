import warnings
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
# from torchvision.models.detection.rpn import AnchorGenerator
# from .rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer
from torch.jit.annotations import List, Optional, Dict, Tuple


class URCNN(nn.Module):
    """
        Implements U-RCNN.

        The input image to the model1 is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

        The behavior of the model1 changes depending if it is in training or evaluation mode.

        During training, the model1 expects both the input tensor, as well as a target (dictionary),
        containing:
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
              between 0-H and 0-W
            - labels (Int64Tensor[N]): the class label for each ground-truth box
            - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

        The model1 returns a Dict[Tensor], containing the classification and regression losses
        for both the RPN and the R-CNN, and the mask loss.

        During inference, the model1 requires only the input tensor, and returns the post-processed
        predictions as a Dict[Tensor]. The fields of the Dict are as
        follows:
            - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format,
              with values between 0-H and 0-W
            - labels (Int64Tensor[N]): the predicted labels
            - scores (FloatTensor[N]): the scores for each prediction
            - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
              obtain the final segmentation masks, the soft masks can be thresholded, generally
              with a value of 0.5 (mask >= 0.5)

        Arguments:
            backbone (nn.Module): the network used to compute the features for the model1.
            num_classes (int): number of output classes of the model1 (including the background).

            rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
                considered as positive during training of the RPN.
            rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
                considered as negative during training of the RPN.
            rpn_num_samples (int): number of anchors that are sampled during training of the RPN
                for computing the loss
            rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
            rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
                bounding boxes
            rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
            rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
            rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
            rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
            rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

            box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
                considered as positive during training of the classification head
            box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
                considered as negative during training of the classification head
            box_num_samples (int): number of proposals that are sampled during training of the
                classification head
            box_positive_fraction (float): proportion of positive proposals during training of the
                classification head
            box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
                bounding boxes
            box_score_thresh (float): during inference, only return proposals with a classification score
                greater than box_score_thresh
            box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
            box_num_detections (int): maximum number of detections, for all classes.

        """

    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        # ------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_num_samples, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # ------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = RoIHeads(
            box_roi_pool, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_num_samples, box_positive_fraction,
            box_reg_weights,
            box_score_thresh, box_nms_thresh, box_num_detections)

        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)

        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        # ------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1024,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])

        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        _targets = []
        for i, _ in enumerate(images):
            _targets.append({k: v[i] for k, v in targets.items()})
        images, targets = self.transformer(images, _targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, _targets)
        proposals = torch.stack(proposals, dim=0)
        print(proposals.shape)
        detections, detector_losses = self.head(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if(images.shape[0] == 1)
        # ori_image_shape = image.shape[-2:]
        #
        # # image, target = self.transformer(image, target)
        # image_shape = image.shape[-2:]
        #
        # feature = self.backbone(image)
        # proposal, rpn_losses = self.rpn(image, feature, image_shape, target)
        # result, roi_losses = self.head(feature, proposal, image_shape, target)
        #
        # if self.training:
        #     return dict(**rpn_losses, **roi_losses)
        # else:
        #     result = self.transformer.postprocess(result, image_shape, ori_image_shape)
        #     return result


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """

        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        # body = models.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=None)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # remove fully connected layers
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256

        # in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)
            # npx = x.cpu().numpy()
            # print(module, x.max(), x.shape)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x


# pretrained=True, num_classes=num_classes
def URCNN_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model1 with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model1 pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """

    if pretrained:
        backbone_pretrained = False

    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = URCNN(backbone, num_classes)

    if pretrained:
        try:
            model_state_dict = torch.load("/home/zhucc/stomata_index/U-RCNN/pretrained_models/resnet50-19c8e357.pth")
            print("loading pretrained_backbone from resnet50-19c8e357.pth ...")
        except Exception:
            print("loading pretrained_backbone from https://download.pytorch.org ...")
            model_urls = {
                'maskrcnn_resnet50_fpn_coco': 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
            }
            model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])

        pretrained_msd = list(model_state_dict.values())
        # del_list = [i for i in range(265, 271)] + [i for i in range(273, 278)]
        # for i, del_idx in enumerate(del_list):
        #     pretrained_msd.pop(del_idx - i)

        pretrained_msd.pop()
        pretrained_msd.pop()
        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            # if i in skip_list:
            # print(name)
            if i >= len(pretrained_msd):
                continue
            try:
                if 'running_var' in name:
                    pretrained_msd[i] = torch.clamp(pretrained_msd[i], min=0.0001)
                    # print(name, i, pretrained_msd[i])
                msd[name].copy_(pretrained_msd[i])
            except Exception:
                print(name, 'not match')
        model.load_state_dict(msd)

    return model
