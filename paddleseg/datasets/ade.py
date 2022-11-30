# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F

URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


@manager.DATASETS.add_component
class ADE20K(Dataset):
    """
    ADE20K dataset `http://sceneparsing.csail.mit.edu/`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 150

    def __init__(self,
                 transforms,
                 dataset_root=None,
                 mode='train',
                 edge=False,
                 to_mask=False,
                 size_divisibility=0,
                 normalize=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge
        self.to_mask = to_mask
        self.size_divisibility = size_divisibility
        self.normalize = normalize

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in ADE20K dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME,
                extraname='ADEChallengeData2016')
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        if mode == 'train':
            img_dir = os.path.join(self.dataset_root, 'images/training')
            label_dir = os.path.join(self.dataset_root, 'annotations/training')
        elif mode == 'val':
            img_dir = os.path.join(self.dataset_root, 'images/validation')
            label_dir = os.path.join(self.dataset_root,
                                     'annotations/validation')
        img_files = os.listdir(img_dir)
        label_files = [i.replace('.jpg', '.png') for i in img_files]
        for i in range(len(img_files)):
            img_path = os.path.join(img_dir, img_files[i])
            label_path = os.path.join(label_dir, label_files[i])
            self.file_list.append([img_path, label_path])

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.file_list[idx]

        data['img'] = image_path
        data['gt_fields'] = [
        ]  # If key in gt_fields, the data[key] have transforms synchronous.

        if self.mode == 'val':
            data = self.transforms(data)
            label = np.asarray(Image.open(label_path))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            label = label[np.newaxis, :, :]
            data['label'] = label - 1
            # data['img'] = paddle.to_tensor(data['img'])
        else:
            data['label'] = label_path
            data['gt_fields'].append('label')
            data = self.transforms(data)
            data['label'] = data['label'] - 1
            # Recover the ignore pixels adding by transform
            data['label'][data['label'] == 254] = 255
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask

            if self.size_divisibility > 0:
                image_size = (data['img'].shape[-2], data['img'].shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                data['img'] = nn.functional.pad(
                    paddle.to_tensor(data['img']).unsqueeze(0),
                    padding_size,
                    value=128).squeeze(0)

                if data['label'] is not None:
                    data['label'] = nn.functional.pad(
                        paddle.to_tensor(
                            data['label'],
                            dtype='int64').unsqueeze(0).unsqueeze(0),
                        padding_size,
                        value=self.ignore_index).squeeze(0).squeeze(0).numpy()

        if self.normalize:
            data['img'] = data['img'].cast('float32')
            mean = paddle.to_tensor(np.array([123.675, 116.280, 103.530
                                              ])).reshape([3, 1, 1])
            std = paddle.to_tensor(np.array([58.395, 57.120, 57.375])).reshape(
                [3, 1, 1])
            data['img'] -= mean
            data['img'] /= std
            data['img'] = data['img'].numpy()

        if self.to_mask:
            sem_seg_gt = data['label']
            instances = {"image_shape": data['img'].shape[1:]}
            classes = np.unique(sem_seg_gt)
            classes = classes[classes != self.ignore_index]

            # To make data compat with dataloader
            classes_cpt = np.array([
                self.ignore_index
                for i in range(self.num_classes - len(classes))
            ])
            classes_cpt = np.append(classes, classes_cpt)
            instances["gt_classes"] = paddle.to_tensor(
                classes_cpt, dtype="int64")

            masks = []
            for cid in classes:
                masks.append(sem_seg_gt == cid)  # [C, H, W] 

            shape = [self.num_classes - len(masks)] + list(data['label'].shape)
            masks_cpt = paddle.zeros(shape=shape)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances['gt_masks'] = paddle.zeros(
                    (150, sem_seg_gt.shape[-2],
                     sem_seg_gt.shape[-1]))  #150, 512, 512

            else:
                instances['gt_masks'] = paddle.concat(
                    [
                        paddle.stack([
                            paddle.cast(
                                paddle.to_tensor(
                                    np.ascontiguousarray(x.copy())), "float32")
                            for x in masks
                        ]), masks_cpt
                    ],
                    axis=0)

            data['instances'] = instances

        return data


if __name__ == "__main__":
    d = ADE20K([], '/ssd2/tangshiyu/PaddleSeg/data/ADEChallengeData2016/')
    for i in range(len(d)):
        # print(data)
        data = d[i]
