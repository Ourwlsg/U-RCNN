import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import pycocotools.mask as mask_util
from torchvision import transforms
from torchvision.transforms import transforms
from torch.utils.data.dataloader import default_collate
from .generalized_dataset import GeneralizedDataset

# VOC_CLASSES = (
#     "aeroplane", "bicycle", "bird", "boat", "bottle",
#     "bus", "car", "cat", "chair", "cow", "diningtable",
#     "dog", "horse", "motorbike", "person", "pottedplant",
#     "sheep", "sofa", "train", "tvmonitor"
# )

VOC_CLASSES = ("stomata",)


# return image, target(image_id=img_id, boxes=boxes, labels=labels, masks=mask)


# image=image, image_id=img_id, boxes=boxes, labels=labels, masks=mask
class BatchCollator:
    def __init__(self, train=True):
        self.is_train = train

    def __call__(self, batch):
        # images = [torch.from_numpy(b['image']) for b in batch]
        images = [b['image'] for b in batch]

        targets = [b['target'] for b in batch]
        bboxes = [target['boxes'] for target in targets]
        labels = [target['labels'] for target in targets]

        max_num_bboxes = max(annot.shape[0] for annot in bboxes)
        if max_num_bboxes > 0:
            for idx, bbox in enumerate(bboxes):
                bbox_padded = torch.ones(max_num_bboxes, 4) * -1
                if bbox.shape[0] > 0:
                    bbox_padded[:bbox.shape[0], 0:4] = torch.from_numpy(bbox)
                    targets[idx]['boxes'] = bbox_padded

            for idx, label in enumerate(labels):
                if label.shape[0] > 0:
                    label_padded = torch.ones(max_num_bboxes) * -1
                    label_padded[:len(label)] = torch.from_numpy(label)
                    targets[idx]['labels'] = label_padded
        else:
            for idx, bbox in enumerate(bboxes):
                if bbox.shape[0] > 0:
                    bbox_padded = torch.ones((len(bboxes), 1, 4)) * -1
                    targets[idx]['boxes'] = bbox_padded
            for idx, label in enumerate(labels):
                if label.shape[0] > 0:
                    label_padded = torch.ones((len(labels), 1)) * -1
                    targets[idx]['labels'] = label_padded

        images = torch.from_numpy(np.stack(images, axis=0)).float()
        # images = images.permute(0, 3, 1, 2)

        # targets = default_collate(targets)
        # targets = torch.from_numpy(np.stack(targets, axis=0))
        return images, targets

        # images, image_ids, boxess, labelss, maskss = zip(*batch)
        # print(len(images), len(image_ids), len(boxess), len(labelss), len(maskss))

        # images = default_collate(images)
        # img_ids = default_collate(image_ids)
        # assert len(images) == len(img_ids)
        #
        # boxes = None
        # labels = None
        # masks = None
        #
        # pad_boxes_list = []
        # pad_labels_list = []
        # batch_size = len(images)
        # if self.is_train:
        #     max_num = 0
        #     for i in range(batch_size):
        #         n = boxess[i].shape[0]
        #         if n > max_num:
        #             max_num = n
        #     for i in range(batch_size):
        #         pad_boxes_list.append(
        #             F.pad(boxess[i], pad=(0, 0, 0, max_num - boxess[i].shape[0]), value=-1))
        #         pad_labels_list.append(
        #             F.pad(labelss[i], pad=(0, max_num - labelss[i].shape[0]), value=-1))
        #
        #         boxes = default_collate(pad_boxes_list)
        #         labels = default_collate(pad_labels_list)
        #         masks = default_collate(maskss)
        #
        # targets = dict(image_ids=img_ids, boxes=boxes, labels=labels, masks=masks)


# dataset_train = VOCDataset(args.data_dir, "train2017", train=True)
class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, ids, transform, train=False):
        super().__init__(ids, max_workers=4, verbose=False)
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        # # instances segmentation task
        # id_file = os.path.join(data_dir, "ImageSets/Segmentations/{}.txt".format(split))
        # self.ids = [id_.strip() for id_ in open(id_file)]
        # self.id_compare_fn = lambda x: int(x.replace("_", ""))
        #
        # self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split))
        # self._coco = None
        #
        self.classes = VOC_CLASSES
        # # resutls' labels convert to annotation labels
        # self.ann_labels = {self.classes.index(n): i for i, n in enumerate(self.classes)}
        #
        # checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split))
        # if train:
        #     if not os.path.exists(checked_id_file):
        #         self.make_aspect_ratios()
        #     self.check_dataset(checked_id_file)

    def make_aspect_ratios(self):
        self._aspect_ratios = []
        for img_id in self.ids:
            anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
            size = anno.findall("size")[0]
            width = size.find("width").text
            height = size.find("height").text
            ar = int(width) / int(height)
            self._aspect_ratios.append(ar)

    def get_image(self, img_id):
        image = cv2.imread(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id.strip())), cv2.IMREAD_COLOR)
        img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img_RGB)
            image = augmented["image"]

        image = transforms.ToTensor()(image)
        # return image.permute(1, 2, 0)
        return image

    def get_target(self, img_id):
        mask = cv2.imread(os.path.join(self.data_dir, "Masks/{}.jpg".format(img_id.strip())), cv2.IMREAD_GRAYSCALE)
        # mask_RGB = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # maskParser
        # mask = transforms.ToTensor()(mask).to(torch.uint8)

        if self.transform is not None:
            augmented = self.transform(image=mask)
            mask = augmented["image"]

        if mask.max() > 1:
            mask = mask / 255.0
        mask[mask >= 0.6] = 1
        mask[mask < 0.6] = 0
        mask = transforms.ToTensor()(mask)

        # annoParser
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id.strip())))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = self.classes.index(name)

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = np.asarray(boxes)
        labels = np.asarray(labels)

        img_id = torch.tensor([self.ids.index(img_id)])
        # mask = torch.from_numpy(mask).float().permute(2, 0, 1)
        mask = mask.float()
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=mask)
        return target

    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO
            self.convert_to_coco_format()
            self._coco = COCO(self.ann_file)
        return self._coco

    def convert_to_coco_format(self, overwrite=False):
        if overwrite or not os.path.exists(self.ann_file):

            print("Generating COCO-style annotations...")
            voc_dataset = VOCDataset(self.data_dir, self.split, True)
            instances = defaultdict(list)
            instances["categories"] = [{"id": i, "name": n} for i, n in enumerate(voc_dataset.classes)]

            ann_id_start = 0
            for image, target in voc_dataset:
                image_id = target["image_id"].item()

                filename = voc_dataset.ids[image_id] + ".jpg"
                h, w = image.shape[-2:]
                img = {"id": image_id, "file_name": filename, "height": h, "width": w}
                instances["images"].append(img)

                anns = target_to_coco_ann(target)
                for ann in anns:
                    ann["id"] += ann_id_start
                    instances["annotations"].append(ann)
                ann_id_start += len(anns)

            json.dump(instances, open(self.ann_file, "w"))
            print("Created successfully: {}".format(self.ann_file))


def target_to_coco_ann(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    area = boxes[:, 2] * boxes[:, 3]
    area = area.tolist()
    boxes = boxes.tolist()

    rles = [
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    anns = []
    for i, rle in enumerate(rles):
        anns.append(
            {
                'image_id': image_id,
                'id': i,
                'category_id': labels[i],
                'segmentation': rle,
                'bbox': boxes[i],
                'area': area[i],
                'iscrowd': 0,
            }
        )
    return anns
