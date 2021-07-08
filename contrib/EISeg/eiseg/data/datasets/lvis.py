import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data.base import ISDataset
from data.sample import DSample


class LvisDataset(ISDataset):
    def __init__(self, dataset_path, split='train',
                 max_overlap_ratio=0.5,
                 **kwargs):
        super(LvisDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        train_categories_path = dataset_path / 'train_categories.json'
        self._train_path = dataset_path / 'train'
        self._val_path = dataset_path / 'val'

        self.split = split
        self.max_overlap_ratio = max_overlap_ratio

        with open( dataset_path / split / f'lvis_{self.split}.json', 'r') as f:
            json_annotation = json.loads(f.read())

        self.annotations = defaultdict(list)
        for x in json_annotation['annotations']:
            self.annotations[x['image_id']].append(x)

        if not train_categories_path.exists():
            self.generate_train_categories(dataset_path, train_categories_path)
        self.dataset_samples = [x for x in json_annotation['images']
                                if len(self.annotations[x['id']]) > 0]

    def get_sample(self, index) -> DSample:
        image_info = self.dataset_samples[index]
        image_id, image_url = image_info['id'], image_info['coco_url']
        image_filename = image_url.split('/')[-1]
        image_annotations = self.annotations[image_id]
        random.shuffle(image_annotations)

        # LVISv1 splits do not match older LVIS splits (some images in val may come from COCO train2017)
        if 'train2017' in image_url:
            image_path = self._train_path / 'images' / image_filename
        else:
            image_path = self._val_path / 'images' / image_filename
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = None
        instances_area = defaultdict(int)
        objects_ids = []
        for indx, obj_annotation in enumerate(image_annotations):
            mask = self.get_mask_from_polygon(obj_annotation, image)
            object_mask = mask > 0
            object_area = object_mask.sum()

            if instances_mask is None:
                instances_mask = np.zeros_like(object_mask, dtype=np.int32)

            overlap_ids = np.bincount(instances_mask[object_mask].flatten())
            overlap_areas = [overlap_area / instances_area[inst_id] for inst_id, overlap_area in enumerate(overlap_ids)
                             if overlap_area > 0 and inst_id > 0]
            overlap_ratio = np.logical_and(object_mask, instances_mask > 0).sum() / object_area
            if overlap_areas:
                overlap_ratio = max(overlap_ratio, max(overlap_areas))
            if overlap_ratio > self.max_overlap_ratio:
                continue

            instance_id = indx + 1
            instances_mask[object_mask] = instance_id
            instances_area[instance_id] = object_area
            objects_ids.append(instance_id)

        return DSample(image, instances_mask, objects_ids=objects_ids)


    @staticmethod
    def get_mask_from_polygon(annotation, image):
        mask = np.zeros(image.shape[:2], dtype=np.int32)
        for contour_points in annotation['segmentation']:
            contour_points = np.array(contour_points).reshape((-1, 2))
            contour_points = np.round(contour_points).astype(np.int32)[np.newaxis, :]
            cv2.fillPoly(mask, contour_points, 1)

        return mask

    @staticmethod
    def generate_train_categories(dataset_path, train_categories_path):
        with open(dataset_path / 'train/lvis_train.json', 'r') as f:
            annotation = json.load(f)

        with open(train_categories_path, 'w') as f:
            json.dump(annotation['categories'], f, indent=1)
