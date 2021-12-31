import os
import time
import yaml
import torch
import logging
import torchvision
import numpy as np
import torch.nn as nn

from PIL import Image
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src import Meter, datasets, set_seed, K_FOLD
from src.urcnn.backbone.myResnet import resnet34
from src.urcnn.urcnn import URCNN
from src.urcnn.rpn import AnchorGenerator
from src.datasets.voc_dataset import BatchCollator

from albumentations import (
    Compose, Resize, BboxParams
)

if __name__ == "__main__":
    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
                        filename=f"{args.log_dir}/{time.time()}_log.txt")

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    device_ids = range(torch.cuda.device_count())
    logging.info(f'Using device {device} {args.gpus}')
    print(f'Using device {device} {args.gpus}')

    # Albumentations
    train_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        # RandomRotate90(0.5),
        # Flip(p=0.5),
        # ShiftScaleRotate(p=0.2, interpolation=Image.NEAREST),  # , border_mode=cv2.BORDER_CONSTANT, value=0
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    # Albumentations
    valid_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        # ToGray(p=1),
        # Normalize(mean=norm_mean, std=norm_std),
        # # ToTensorV2()
    ], p=1, bbox_params=BboxParams(format='pascal_voc', label_fields=["labels"]))

    # gray
    # norm_mean = [0.46152964]
    # norm_std = [0.10963361]

    # RGB
    norm_mean = [0.4976264, 0.45133978, 0.3993562]
    norm_std = [0.11552592, 0.10886826, 0.10727626]

    # for fold in range(args.K_FOLD):
    for fold in range(1, 2):
        writer = SummaryWriter(log_dir=args.board_dir + f"/fold_{fold}",
                               comment=f'FOLD_{fold}/{args.lr}_BS_{args.batch_size}_EPOCHS_{args.epochs}_inputsize_{args.inputsize}')
        train_ids, val_ids = K_FOLD(os.path.join(args.data_dir, f'ImageSets/fold_{fold}'))
        logging.info(f"{len(train_ids)} images for training")
        logging.info(f"{len(val_ids)} images for validation")

        dataset_train = datasets(args.dataset, data_dir=args.data_dir, ids=train_ids, transform=train_Transform,
                                 train=True)
        train_loader = DataLoader(dataset=dataset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=BatchCollator(train=dataset_train.train),
                                  drop_last=True)

        # 加载URCNN
        # backbone = UNet(n_channels=3, n_classes=1, bilinear=True)
        # backbone = UNetWithResnet50Encoder(n_classes=1)
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
                      box_roi_pool=roi_pooler).to(device)

        # for index, child in enumerate(model.children()):
        #     print(index, child)

        if args.load:
            model.load_state_dict(torch.load(args.load_name, map_location=device), strict=False)
            logging.info(f'Model loaded from {args.load_name}')

        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = {
            "sgd": lambda: torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-8),
            "adam": lambda: torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8, eps=1e-08),
            "rmsprop": lambda: torch.optim.RMSprop(params, lr=args.lr, momentum=0.9, eps=0.001, weight_decay=1e-8)
        }[args.optim]()

        # lr_lambda = lambda x: 0.1 ** bisect.bisect([22, 26], x)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs * len(dataloader), eta_min=0, last_epoch=-1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)
        scheduler = LambdaLR(optimizer, lambda x: (((1 + np.cos(x * np.pi / args.epochs)) / 2) ** 1.0) * 0.9 + 0.1)

        for epoch in range(args.start_epoch, args.epochs + 1):
            iters = len(train_loader) if args.iters < 0 else args.iters

            t_m = Meter("total")
            m_m = Meter("model")
            b_m = Meter("backward")
            model.train()
            A = time.time()
            for i, (images, targets) in enumerate(train_loader):
                T = time.time()
                num_iters = (epoch - 1) * len(train_loader) + i
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
                    # target_gpu = {k: v.to(device).int64() if k=="lables" else k: v.to(device) for k, v in target.items()}
                    targets_gpu.append(target_gpu)

                S = time.time()

                losses, mask_preds = model(images, targets_gpu)
                # if backbone.n_classes > 1:
                #     criterion = nn.CrossEntropyLoss()
                # else:
                #     criterion = nn.BCEWithLogitsLoss()
                criterion = nn.BCEWithLogitsLoss()
                mask_loss = {}
                if model.training:
                    target_masks = []
                    for target in targets:
                        target_masks.append(target["masks"])
                    if len(target_masks) > 1:
                        target_masks_tensor = torch.stack(target_masks, dim=0)
                    else:
                        target_masks_tensor = target_masks[0].unsqueeze(dim=0)
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
                    print("{}\t".format("num_iters"),
                          "\t".join("{}".format(l) for l in losses.keys()),
                          "{}\t".format("total_loss"))
                    print("{}\t".format(num_iters),
                          "\t".join("{:.3f}".format(l.item()) for l in losses.values()),
                          "{}\t".format(total_loss))
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)
                    # scheduler.step()

                    info = {
                        'loss': total_loss,
                    }
                    info.update(losses)
                    writer.add_scalars("losses".format(fold), info, num_iters)
                    writer.add_scalar("lr".format(fold), optimizer.param_groups[0]['lr'], num_iters)

                t_m.update(time.time() - T)
                if i >= iters - 1:
                    break

            scheduler.step()

            if epoch % args.checkpoint_epoch_interval == 0:
                os.makedirs(args.checkpoint_dir + f"/fold{fold}/", exist_ok=True)
                if len(device_ids) > 1:
                    torch.save(model.module.state_dict(),
                               args.checkpoint_dir + f"/fold{fold}/" + f'CP_epoch{epoch}.pth')
                else:
                    torch.save(model.state_dict(),
                               args.checkpoint_dir + f"/fold{fold}/" + f'CP_epoch{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')
            A = time.time() - A
            print(
                "iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg,
                                                                                      1000 * m_m.avg, 1000 * b_m.avg))
