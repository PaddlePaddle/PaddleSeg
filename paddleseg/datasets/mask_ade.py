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

import numpy as np
from PIL import Image

import paddle

from paddleseg.datasets import ADE20K
from paddleseg.cvlibs import manager
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class MaskedADE20K(ADE20K):
    """
    ADE20K dataset `http://sceneparsing.csail.mit.edu/` for Maskformer.
    It returns an additional masked gt for each instance.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        super().__init__(transforms, dataset_root, mode, edge)

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.file_list[idx]

        data['img'] = image_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []

        if self.mode == 'val':
            data = self.transforms(data)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            data['label'] = label - 1
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

        #######################################
        # transform the data into masked data.#
        #######################################
        sem_seg_gt = data['label']
        instances = {"image_shape": data['img'].shape[1:]}
        classes = np.unique(sem_seg_gt)
        classes = classes[classes != self.ignore_index]

        # To make data compatible with dataloader
        classes_cpt = np.array([
            self.ignore_index for i in range(self.num_classes - len(classes))
        ])
        classes_cpt = np.append(classes, classes_cpt)
        instances["gt_classes"] = paddle.to_tensor(classes_cpt, dtype="int64")

        masks = []
        for cid in classes:
            masks.append(sem_seg_gt == cid)  # [C, H, W] 

        shape = [self.num_classes - len(masks)] + list(data['label'].shape)
        masks_cpt = paddle.zeros(shape=shape)

        if len(masks) == 0:
            # Some image does not have annotation will be all ignored
            instances['gt_masks'] = paddle.zeros(
                (150, sem_seg_gt.shape[-2],
                 sem_seg_gt.shape[-1]))  #150, 512, 512

        else:
            instances['gt_masks'] = paddle.concat(
                [
                    paddle.stack([
                        paddle.cast(
                            paddle.to_tensor(np.ascontiguousarray(x.copy())),
                            "float32") for x in masks
                    ]), masks_cpt
                ],
                axis=0)

        data['instances'] = instances

        return data
