import cv2
import json
import random
import numpy as np
from pathlib import Path
from data.base import ISDataset
from data.sample import DSample


class CocoDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob

        self.load_samples()

    def load_samples(self):
        annotation_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}'
        self.images_path = self.dataset_path / self.split

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = annotation['annotations']

        self._categories = annotation['categories']
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)
        self._stuff_labels_set = set(self._stuff_labels)

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])
        label_path = self.labels_path / dataset_sample['file_name']

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        # 这个是什么处理呢
        label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

        instance_map = np.full_like(label, 0)
        things_ids = []
        stuff_ids = []

        for segment in dataset_sample['segments_info']:
            class_id = segment['category_id']
            obj_id = segment['id']
            if class_id in self._things_labels_set:
                if segment['iscrowd'] == 1:
                    continue
                things_ids.append(obj_id)
            else:
                stuff_ids.append(obj_id)

            instance_map[label == obj_id] = obj_id

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            instances_ids = things_ids + stuff_ids
        else:
            instances_ids = things_ids

            for stuff_id in stuff_ids:
                instance_map[instance_map == stuff_id] = 0

        return DSample(image, instance_map, objects_ids=instances_ids)

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')
