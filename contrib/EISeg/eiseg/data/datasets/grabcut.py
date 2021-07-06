from pathlib import Path

import cv2
import numpy as np

from data.base import ISDataset
from data.sample import DSample


class GrabCutDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 **kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)
