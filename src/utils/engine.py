import os
import time

import torch

from src.utils.util import K_FOLD, Meter

try:
    from src.datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()

        losses = model(image, target)
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
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg,
                                                                                1000 * m_m.avg, 1000 * b_m.avg))
    return A / iters



# generate results file   
@torch.no_grad()
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.ann_labels

    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)

        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg))

    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, args.results)

    return A / iters





if __name__ == "__main__":
    import torchvision
    import yaml
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from easydict import EasyDict
    from torch.utils.data.dataloader import DataLoader

    from src import datasets
    from src.datasets.voc_dataset import BatchCollator
    from src.urcnn.rpn import AnchorGenerator
    from src.urcnn.backbone.myResnet import resnet34
    from src.urcnn.urcnn import URCNN

    from albumentations import Compose, Resize, BboxParams

    with open("../../cfgs/test.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    norm_mean = [0.4976264, 0.45133978, 0.3993562]
    norm_std = [0.11552592, 0.10886826, 0.10727626]
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    backbone = resnet34(3, 1, True)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=((4, 8, 16),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = URCNN(backbone,
                  num_classes=args.num_classes,
                  min_size=args.inputsize,
                  max_size=args.inputsize,
                  image_mean=norm_mean,
                  image_std=norm_std,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_classes=2).to(device)

    # for index, child in enumerate(model.children()):
    #     print(index, child)

    valid_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{1}'))

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=False)
    dataset_valid = datasets(args.dataset,
                             train=False,
                             ids=val_ids,
                             data_dir=args.data_dir,
                             transform=valid_Transform)
    valid_loader = DataLoader(shuffle=False,
                              num_workers=0,
                              drop_last=False,
                              pin_memory=True,
                              dataset=dataset_valid,
                              batch_size=args.batch_size,
                              collate_fn=BatchCollator(train=dataset_valid.train))

    model.eval()

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

    iters = len(valid_loader) if args.iters < 0 else args.iters
    all_boxes = [[[] for _ in range(val_ids)] for _ in range(args.num_classes)]
    for i, (images, targets) in enumerate(valid_loader):
        images = images.to(device)
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
            targets_gpu.append(target_gpu)

        object_preds, mask_preds = model(images, targets_gpu)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        mask_preds = mask_preds.permute(0, 2, 3, 1).cpu()





        plt.figure(figsize=(20, 10))
        for index, image1 in enumerate(images):
            ax1 = plt.subplot(2, 4, index + 1)
            image3 = (image1 * 255).astype(np.uint8).copy()
            # image3 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            bboxes = object_preds[index]["boxes"].cpu()
            scores = object_preds[index]["scores"].cpu()
            for i, bbox in enumerate(object_preds[index]["boxes"].cpu()):
                bbox = bbox.detach().numpy()
                cv2.rectangle(image3, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0),
                              thickness=2)
                cv2.putText(image3, '%.3f' % scores[i], (int(bbox[0]), int(bbox[1]) + 15), cv2.FONT_HERSHEY_PLAIN,
                            0.5, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            ax1.imshow(image3)
            ax2 = plt.subplot(2, 4, index + 5)

            mask = torch.sigmoid(mask_preds[index]).detach().numpy()
            ax2.imshow(mask * 255)
        plt.show()
