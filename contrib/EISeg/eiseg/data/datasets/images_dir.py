import cv2
import numpy as np
from pathlib import Path

from data.base import ISDataset
from data.sample import DSample


class ImagesDirDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='images', masks_dir_name='masks',
                 **kwargs):
        super(ImagesDirDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        images_list = [x for x in sorted(self._images_path.glob('*.*'))]

        samples = {x.stem: {'image': x, 'masks': []} for x in images_list}
        for mask_path in self._insts_path.glob('*.*'):
            mask_name = mask_path.stem
            if mask_name in samples:
                samples[mask_name]['masks'].append(mask_path)
                continue

            mask_name_split = mask_name.split('_')
            if mask_name_split[-1].isdigit():
                mask_name = '_'.join(mask_name_split[:-1])
                assert mask_name in samples
                samples[mask_name]['masks'].append(mask_path)

        for x in samples.values():
            assert len(x['masks']) > 0, x['image']

        self.dataset_samples = [v for k, v in sorted(samples.items())]

    def get_sample(self, index) -> DSample:
        sample = self.dataset_samples[index]
        image_path = str(sample['image'])

        objects = []
        ignored_regions = []
        masks = []
        for indx, mask_path in enumerate(sample['masks']):
            gt_mask = cv2.imread(str(mask_path))[:, :, 0].astype(np.int32)
            instances_mask = np.zeros_like(gt_mask)
            instances_mask[gt_mask == 128] = 2
            instances_mask[gt_mask > 128] = 1
            masks.append(instances_mask)
            objects.append((indx, 1))
            ignored_regions.append((indx, 2))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return DSample(image, np.stack(masks, axis=2),
                       objects_ids=objects, ignore_ids=ignored_regions, sample_id=index)