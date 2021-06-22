from pathlib import Path

import os
import cv2
import numpy as np

from data.base import ISDataset
from data.sample import DSample


class HumanDataset(ISDataset):
    def __init__(self, dataset_path,
                 split = 'train',
                 **kwargs):
        super(HumanDataset, self).__init__(**kwargs)
        
        self.mode = split.lower()
        self.path = dataset_path

        if self.mode == 'train':
            file_path = os.path.join(self.path, 'train_mini.txt')
        else:
            file_path = os.path.join(self.path, 'val_mini.txt')

        self.dataset_samples = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.dataset_samples.append(line)

    def get_sample(self, index):
        items = self.dataset_samples[index].split(' ')
        if 'person_detection__ds' in items[0]:
            image_path, image_name = items[0].rsplit('/', 1)
            items[0] = image_path.rsplit('/', 1)[0] + '/' + image_name
            mask_path, mask_name = items[1].rsplit('/', 1)
            items[1] = mask_path.rsplit('/', 1)[0] + '/' + mask_name
        

        image_path = os.path.join(self.path, items[0])
        mask_path = os.path.join(self.path, items[1])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1
       
        return  DSample(image, instances_mask, objects_ids=[1], sample_id=index)

