import logging
import os
import cv2
import pickle
import torch
from tqdm import tqdm
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np

from src.utils.gpu import toDevice


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        input = torch.sigmoid(input)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t


def dice_coeff(pre, gt):
    """Dice coeff for batches"""
    if pre.is_cuda:
        sum_ = torch.FloatTensor(1).cuda().zero_()
    else:
        sum_ = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(pre, gt)):
        s = DiceCoeff().forward(c[0], c[1])
        sum_ = sum_ + s

    return sum_ / (i + 1)


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def write_val_bbox_gt(valid_loader, cachefile, classname):
    val_bbox_gt = {}
    for i, (images, targets) in tqdm(enumerate(valid_loader), desc="writing GT"):
        bboxes_gt = targets[0]["boxes"].numpy()
        image_id = targets[0]["image_id"]
        objects = []
        for bbox in bboxes_gt:
            obj_struct = {'name': classname, 'difficult': 0, 'bbox': bbox}
            objects.append(obj_struct)

        val_bbox_gt[image_id] = objects
    logging.info('Saving cached annotations to {:s}'.format(cachefile))
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(val_bbox_gt, f)


def read_val_bbox_gt(cacheFile):
    with open(cacheFile, 'rb') as f:
        try:
            recs = pickle.load(f)
        except:
            recs = pickle.load(f, encoding='bytes')
    return recs


def write_val_bbox_dl(filename, ids, all_boxes_scores):
    logging.info(f'Writing bbox results to {filename}')
    print(f'Writing bbox results to {filename}')
    with open(filename, 'wt') as file:
        for index, image_id in enumerate(ids):
            score = all_boxes_scores[1][index]
            dets = all_boxes_scores[0][index]
            if len(dets) == 0:
                continue
            for k in range(len(dets)):
                file.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                           format(image_id.strip(),
                                  score[k],
                                  dets[k][0],
                                  dets[k][1],
                                  dets[k][2],
                                  dets[k][3]))


def read_val_bbox_dl(detfile):
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.split(' ') for x in lines]
    image_ids = [x[0].strip() for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    Bboxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

    return Bboxes, image_ids, confidence


def get_recal_precision(Bboxes, image_ids, confidence, class_recs, num_positive):
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if Bboxes.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        Bboxes = Bboxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bbox = Bboxes[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bbox[0])
                iymin = np.maximum(BBGT[:, 1], bbox[1])
                ixmax = np.minimum(BBGT[:, 2], bbox[2])
                iymax = np.minimum(BBGT[:, 3], bbox[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                union = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
                         (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                         (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / union
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > 0.5:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_positive)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec


def draw_PRcurve(rec, prec, ap, save_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(rec, prec, lw=2, label='(AP = {:.4f})'.format(ap))
    with open(os.path.join(save_dir, f'rec_prec_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(f'{save_dir}/PR.jpg')
    plt.close()
    # plt.show()


def validation(fold, epoch, val_ids, valid_loader, model, device, args, tfwriter):
    cacheDir = f"{args.cache_dir}/fold-{fold}"
    os.makedirs(cacheDir, exist_ok=True)
    cacheFile = os.path.join(cacheDir, f"bbox_GT_height-{args.height}_width-{args.width}cache.pkl")

    # write bboxGT
    if not os.path.isfile(cacheFile):
        write_val_bbox_gt(valid_loader, cacheFile, "stomata")

    num_val = len(valid_loader)
    all_boxes_scores_dp = [[[] for _ in range(num_val)] for _ in range(2)]
    sum_dice = 0
    model.eval()
    # loss_mask = 0
    # loss_classifier = 0
    # loss_box_reg = 0
    # loss_objectness = 0
    # loss_rpn_box_reg = 0
    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(valid_loader), desc="validating"):
            images = images.to(device)
            # targets_gpu = toDevice(targets, device)
            mask = targets[0]["masks"].to(device)
            object_preds, mask_preds = model(images, None, args.OD, args.SS)
            if args.SS:
                # loss_mask += losses['loss_mask'].item()

                mask_pred = mask_preds[0]
                sum_dice += dice_coeff(mask_pred, mask).item()
            if args.OD:
                # loss_classifier += losses['loss_classifier'].item()
                # loss_box_reg += losses['loss_box_reg'].item()
                # loss_objectness += losses['loss_objectness'].item()
                # loss_rpn_box_reg += losses['loss_rpn_box_reg'].item()
                object_pred = object_preds[0]
                all_boxes_scores_dp[0][i] = object_pred["boxes"].cpu().numpy()
                all_boxes_scores_dp[1][i] = object_pred["scores"].cpu().numpy()

            # stoma_pred, masks_pred, mask
            if i == 1:
                if args.SS:
                    R_mask = mask
                    R_cell_pred = (mask_pred > 0.5) * 1.0
                    if tfwriter is not None:
                        tfwriter.add_image('vis/masks_pred', R_cell_pred, epoch)
                        tfwriter.add_image('vis/mask_gt', R_mask, epoch)

                if args.OD:
                    R_stoma_pred = images[0]
                    idx = torch.nonzero(object_pred["scores"] > args.vis_score_thre)
                    object_pred_thre = torch.index_select(object_pred["boxes"], dim=0, index=idx.flatten())
                    if tfwriter is not None:
                        tfwriter.add_image_with_boxes('vis/stoma_pred',
                                                      R_stoma_pred,
                                                      object_pred_thre,
                                                      epoch)

                        # for bbox in all_boxes_scores_dp[0][i]:
                        #     cv2.rectangle(R_stoma_pred,
                        #                   (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                        #                   color=(1, 0, 0),
                        #                   thickness=1)
        dice = 0
        if args.SS:
            dice = sum_dice / num_val

        ap = 0
        if args.OD:
            # write detections
            detectionsDir = os.path.join(args.output_dir, args.experiment, f"fold-{fold}", f"epoch-{epoch}")
            os.makedirs(detectionsDir, exist_ok=True)
            detectionsFile = os.path.join(detectionsDir, f"detections.txt")
            write_val_bbox_dl(detectionsFile, val_ids, all_boxes_scores_dp)

            # read bboxGT
            logging.info(f"Loading bboxGT from {cacheFile}...")
            print(f"Loading bboxGT from {cacheFile}...")
            recs = read_val_bbox_gt(cacheFile)
            class_recs = {}
            num_positive = 0
            for image_id in val_ids:
                image_id = image_id.strip()
                objs = [obj for obj in recs[image_id] if obj['name'] == "stomata"]
                det = [False] * len(objs)
                bbox = np.array([x['bbox'] for x in objs])
                difficult = np.array([x['difficult'] for x in objs]).astype(bool)
                num_positive = num_positive + sum(~difficult)
                class_recs[image_id] = {'bbox': bbox,
                                        'difficult': difficult,
                                        'det': det}
            # read detections
            Bboxes, image_ids, confidence = read_val_bbox_dl(detectionsFile)
            recall, precision = get_recal_precision(Bboxes, image_ids, confidence, class_recs, num_positive)
            ap = voc_ap(recall, precision, use_07_metric=False)
            draw_PRcurve(recall, precision, ap, detectionsDir)
        # losses_val = {
        #     'loss_mask': loss_mask / num_val,
        #     'loss_box_reg': loss_box_reg / num_val,
        #     'loss_classifier': loss_classifier / num_val,
        #     'loss_objectness': loss_objectness / num_val,
        #     'loss_rpn_box_reg': loss_rpn_box_reg / num_val,
        #     'loss_total': (loss_mask + loss_classifier+ loss_box_reg + loss_objectness + loss_rpn_box_reg) / num_val
        # }
        if tfwriter is not None:
            # tfwriter.add_scalars("losses/val", losses_val, epoch)
            if args.SS:
                tfwriter.add_scalar("val/cell_dice", dice, epoch)
            if args.OD:
                tfwriter.add_scalar("val/stoma_ap", ap, epoch)

        logging.info('~~~~~~~~~~~~~~~~~~~~')
        logging.info('Validation Results:')
        logging.info('~~~~~~~~~~~~~~~~~~~~')
        logging.info('AP   = {:.4f}'.format(ap))
        logging.info('DICE = {:.4f}'.format(dice))
        logging.info('--------------------------------------------------------------')
        logging.info('Results computed with the **unofficial** Python eval code.')
        logging.info('Results should be very close to the official MATLAB eval code.')
        logging.info('--------------------------------------------------------------')
        logging.info('')

        print('~~~~~~~~~~~~~~~~~~~~')
        print('Validation Results:')
        print('~~~~~~~~~~~~~~~~~~~~')
        print('AP   = {:.4f}'.format(ap))
        print('DICE = {:.4f}'.format(dice))
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
        print('')
        return ap, dice


if __name__ == "__main__":
    import torchvision
    import yaml
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from easydict import EasyDict
    from torch.utils.data.dataloader import DataLoader
    from src.utils.util import K_FOLD
    from src import datasets
    from src.datasets.voc_dataset import BatchCollator
    from src.urcnn.rpn import AnchorGenerator
    from src.urcnn.backbone.myResnet import resnet34
    from src.urcnn.urcnn import URCNN
    from albumentations import Compose, Resize, BboxParams
    import warnings

    warnings.filterwarnings("ignore")
    with open("cfgs/test.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    norm_mean = [0.4976264, 0.45133978, 0.3993562]
    norm_std = [0.11552592, 0.10886826, 0.10727626]
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    valid_Transform = Compose([
        Resize(height=args.height, width=args.width, interpolation=Image.NEAREST, p=1),
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    train_val_split_path = os.path.join(args.data_dir, f'ImageSets/fold_{4}')
    train_ids, val_ids = K_FOLD(train_val_split_path)

    dataset_valid = datasets(args.dataset,
                             train=False,
                             ids=train_ids,
                             data_dir=args.data_dir,
                             transform=valid_Transform)
    valid_loader = DataLoader(shuffle=False,
                              num_workers=0,
                              drop_last=False,
                              pin_memory=True,
                              dataset=dataset_valid,
                              batch_size=1,
                              collate_fn=BatchCollator(train=dataset_valid.train))

    backbone = resnet34(3, False)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=((4, 8, 16),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = URCNN(backbone,
                  num_classes=args.num_classes,
                  min_size=args.height,
                  max_size=args.width,
                  image_mean=norm_mean,
                  image_std=norm_std,
                  box_score_thresh=args.train_score_thre,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_classes=2).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=False)
    model.eval()
    ap, dice = validation(4, 100000, train_ids, valid_loader, model, device, args, tfwriter=None)

    # # 定义总参数量、可训练参数量及非可训练参数量变量
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    #
    # # 遍历model.parameters()返回的全局参数列表
    # for param in model.parameters():
    #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量
    #
    # print(f'Total params: {Total_params}')
    # print(f'Trainable params: {Trainable_params}')
    # print(f'Non-trainable params: {NonTrainable_params}')

    """
    Total params: 38963704
    Trainable params: 38963704
    Non-trainable params: 0
    """
