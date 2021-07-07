import random
import pickle

import cv2
import numpy as np
import paddle

import paddleseg.transforms as T
from .points_sampler import MultiPointSampler


def get_unique_labels(x, exclude_zero=False):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()

    if exclude_zero:
        labels = [x for x in labels if x != 0]
    return labels


class ISDataset(paddle.io.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 min_ignore_object_area=10,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.dataset_samples = None
        
    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = x[:,:,None]
        #img = paddle.to_tensor(x.transpose([2,0,1])).astype('float32') / 255
        img = x.transpose([2,0,1]).astype(np.float32) / 255
        return img

    def __getitem__(self, index):
        
#         if self.samples_precomputed_scores is not None:
#             index = np.random.choice(self.samples_precomputed_scores['indices'],
#                                      p=self.samples_precomputed_scores['probs'])
#         else:
#             if self.epoch_len > 0:
#                 index = random.randrange(0, len(self.dataset_samples))
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)
        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points()).astype(np.float32)
        mask = self.points_sampler.selected_mask
        image = self.to_tensor(sample.image)
        ids = sample.sample_id

        return image, points, mask

    def augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores
