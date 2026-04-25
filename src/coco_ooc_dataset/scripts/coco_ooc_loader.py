#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:28:55 2019

@author: manoj
"""
import ipdb
from PIL import Image
import torch
import os
import numpy as np
# from src.coco_ooc_dataset.scripts.voc_loader import imgtransform
from typing import Any, List
from pathlib import Path
from torchvision import transforms as T
import copy
import math

COCO_VOC_CATS = ['__background__', 'airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                 'dog', 'horse', 'motorcycle', 'person', 'potted plant',
                 'sheep', 'couch', 'train', 'tv']

COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

COCO_CATS = COCO_VOC_CATS + COCO_NONVOC_CATS

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
                67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
                38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
                81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
                42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
                80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
                7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
                46, 'zebra': 24}

coco_ids_to_cats = {v: k for k, v in coco_ids.items()}
coco_fake_ids = {coco_ids[k]: i + 1 for i, k in enumerate(sorted(coco_ids))}
coco_fake2real = {v: k for k, v in coco_fake_ids.items()}
coco_fake2names = {fakeid: coco_ids_to_cats[realid] for fakeid, realid in coco_fake2real.items()}


class CenterCropAdaptive(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        h, w = img.shape[-2:]
        if w > h:
            left = (w - h) // 2
            right = left + h
            top = 0
            bottom = h
        elif h > w:
            top = (h - w) // 2
            bottom = top + w
            left = 0
            right = w
        else:
            return img
        
        img_cropped = img[..., top:bottom, left:right]
        return img_cropped

def retbox(bbox, format='xyxy'):
    """A utility function to return box coords asvisualizing boxes."""
    if format == 'xyxy':
        xmin, ymin, xmax, ymax = bbox
    elif format == 'xywh':
        xmin, ymin, w, h = bbox
        xmax = xmin + w - 1
        ymax = ymin + h - 1

    box = np.array([[xmin, xmax, xmax, xmin, xmin],
                    [ymin, ymin, ymax, ymax, ymin]])
    return box.T


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


COCO_OOC_Dataset_OUTPUT_TYPE = tuple[torch.Tensor, str, torch.Tensor, list[str], torch.Tensor] # (image, ooc_cat, ooc_box, inc_cats, inc_boxes)

class COCO_OOC_Dataset():
    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    ids_to_cats = dict(enumerate(COCO_CATS))
    num_classes = len(COCO_CATS)
    categories = COCO_CATS[1:]

    def __init__(self, root, annFile, included=[], img_size=512, **kwargs):

        from pycocotools.coco import COCO

        self.root = root
        self.included_cats = included
        self.oocd_dir = kwargs.get("oocd_dir")
        self.img_size = img_size

        self.transform = T.Compose([
            T.ToTensor(),
            CenterCropAdaptive(),
            T.Resize((self.img_size, self.img_size))
        ])

        self.coco = COCO(annFile)
        self.ids = list(Path(self.oocd_dir).joinpath("annotations").iterdir())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index) -> COCO_OOC_Dataset_OUTPUT_TYPE:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """

        file = np.load(self.ids[index], allow_pickle=True).tolist()
        coco = self.coco
        img_id = file['image_id']
        ooc_entry = file['ooc_annotation']
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        target = coco.loadAnns(ann_ids)
        path = str(self.ids[index]).replace("annotations", "images").replace(".npy", ".jpg") # new image here
        img = pil_loader(path)
        H, W = img.height, img.width
        # ----add ooc here---
        entry = copy.deepcopy(coco.loadAnns(ooc_entry['coco_ann_id'])[0])
        entry['bbox'] = correct_ooc_bbox(ooc_entry['bbox'])  # replace the old since new is resized and location changed
        entry['isooc'] = 1
        target.append(entry)
        # ------------------------
        ann = self.convert(target)
        ann['fname'] = self.ids[index].parts[-1]
        ann["size"] = torch.as_tensor([int(H), int(W)])

        center = (H / 2, W / 2)

        boxes = ann['boxes']

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            x1_ = max(0, min(1, (x1 - center[1]) / min(H, W) + 0.5))
            y1_ = max(0, min(1, (y1 - center[0]) / min(H, W) + 0.5))
            x2_ = max(0, min(1, (x2 - center[1]) / min(H, W) + 0.5))
            y2_ = max(0, min(1, (y2 - center[0]) / min(H, W) + 0.5))
            boxes[i] = torch.tensor([x1_, y1_, x2_, y2_])

        ooc_box = boxes[ann["isooc"]==1][0].numpy()
        inc_boxes = boxes[ann["isooc"]==0].numpy()

        ooc_cat = coco_fake2names[int(ann["labels"][ann["isooc"]==1][0])]
        inc_cats = [coco_fake2names[int(label)] for label in ann["labels"][ann["isooc"]==0]]

        return self.transform(img), ooc_cat, ooc_box, inc_cats, inc_boxes

    def getannotation(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        raise NotImplementedError("getannotation is not implemented yet. Use __getitem__ instead.")
        file = np.load(self.ids[index], allow_pickle=True).tolist()
        coco = self.coco
        img_id = file['image_id']
        ooc_entry = file['ooc_annotation']
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        target = coco.loadAnns(ann_ids)
        path = str(self.ids[index]).replace("annotations", "images").replace(".npy", ".jpg") # new image here
        img = pil_loader(path)
        H, W = img.height, img.width
        # ----add ooc here---
        entry = copy.deepcopy(coco.loadAnns(ooc_entry['coco_ann_id'])[0])
        entry['bbox'] = correct_ooc_bbox(ooc_entry['bbox'])  # replace the old since new is resized and location changed
        
        entry['isooc'] = 1
        target.append(entry)
        # ------------------------
        ann = self.convert(target)
        ann['fname'] = self.ids[index].parts[-1]
        ann["size"] = torch.as_tensor([int(H), int(W)])
        return img, ann, file

    def convert(self, target):
        boxes = []
        classes = []
        area = []
        iscrowd = []
        isooc = []
        for obj in target:
            bbox = obj['bbox']
            xmin, ymin, w, h = bbox
            # is this right?
            bbox = [xmin, ymin, w + xmin - 1, h + ymin - 1]
            cat = obj['category_id']
            cat = coco_fake_ids[cat]
            difficult = int(obj['iscrowd'])
            ooc_status = int(obj.get('isooc', 0)) # default 0
            if self.included_cats == [] or cat in self.included_cats:
                if not difficult:
                    boxes.append(bbox)
                    classes.append(cat)
                    iscrowd.append(difficult)
                    isooc.append(ooc_status)
                    area.append(w * h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes).long()
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)
        isooc = torch.as_tensor(isooc)
        image_id = obj['image_id']
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["isooc"] = isooc #add ooc status
        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

def correct_ooc_bbox(bbox):
    xmin, ymin, w, h = bbox
    return [ymin, xmin, w, h]

class COCO_OCC_Dataset_dataloader:
    def __init__(
            self, 
            dataset: COCO_OOC_Dataset, 
            batch_size: int, 
            target_classes: List[str]|None=None,
            dataset_step_size: float|None=200,
            max_inc_pr_ooc: int|None=5,
            ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.classes = set(target_classes) if target_classes is not None else set(COCO_CATS)
        self.dataset_step_size = dataset_step_size

        self.max_inc_pr_ooc = max_inc_pr_ooc
        self.generator = self.get_generator()
    
    def reset_generator(self):
        self.generator = self.get_generator()
    
    def __iter__(self):
        while True:
            imgs = []
            categories = []
            boxes = []
            labels = []
            for i in range(self.batch_size):
                try:
                    img, cat, box, label, dataset_idx = next(self.generator)
                    imgs.append(img)
                    categories.append(cat)
                    boxes.append(box)
                    labels.append(label)
                except StopIteration:
                    break
            if len(imgs) == 0:
                break
            yield torch.stack(imgs), categories, torch.from_numpy(np.stack(boxes)), torch.tensor(labels), dataset_idx


    def get_generator(self):
        N = len(self.dataset)
        step_size = self.dataset_step_size or 1
        for i in range(0, N, step_size):
            img, ooc_cat, ooc_box, inc_cats, inc_boxes = self.dataset[i]
            num_inc = self.max_inc_pr_ooc or len(inc_boxes)
            inc_boxes = inc_boxes[:num_inc]

            if ooc_cat in self.classes:
                yield img, ooc_cat, ooc_box, 1, i
            for j in range(len(inc_boxes)):
                if inc_cats[j] in self.classes:
                    yield img, inc_cats[j], inc_boxes[j], 0, i


def get_dataloader(
    coco_ooc_dataset_root = r'C:\Users\Lauge\Downloads\coco_ooc\coco_ooc_dataset',
    coco_2014_ann_file = r'C:\Users\Lauge\fiftyone\coco-2014\raw\instances_val2014.json',
    batch_size = 64,
    dataset_step_size = 200,
    max_in_context_pr_ooc = 5
):
    dataset = COCO_OOC_Dataset(coco_ooc_dataset_root, coco_2014_ann_file, oocd_dir=coco_ooc_dataset_root)
    dataloader = COCO_OCC_Dataset_dataloader(
        dataset, 
        batch_size=batch_size, 
        target_classes=['person', 'car', 'dog', 'cat', 'bicycle', 'airplane', 'bus', 'train', 'truck'],
        dataset_step_size=dataset_step_size,
        max_inc_pr_ooc=max_in_context_pr_ooc
    )
    return dataloader


# %%
if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import math
    from tqdm import tqdm

    # DATASETS_ROOT = './datasets'
    split = 'val2014'
    root = r'C:\Users\Lauge\Downloads\coco_ooc\coco_ooc_dataset'
    annFile = r'C:\Users\Lauge\fiftyone\coco-2014\raw\instances_val2014.json'
    ld = COCO_OOC_Dataset(root, annFile, oocd_dir=r"C:\Users\Lauge\Downloads\coco_ooc\coco_ooc_dataset")

    batch_size = 64
    loader = COCO_OCC_Dataset_dataloader(
        ld, 
        batch_size=batch_size, 
        target_classes=['person', 'car', 'dog', 'cat', 'bicycle', 'airplane', 'bus', 'train', 'truck'],
        dataset_step_size=200,
        max_inc_pr_ooc=5)
    last_dataset_idx = -1

    for batch in (bar := tqdm(loader, total=len(ld))):
        imgs, ooc_cats, ooc_boxes, labels, dataset_idx = batch
        # Sync tqdm progress with dataset index
        if dataset_idx != last_dataset_idx:
            last_dataset_idx = dataset_idx
            bar.n = min(dataset_idx + 1, len(ld))
            bar.refresh()
