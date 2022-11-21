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

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F

URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

# class Instances(object):
#     """
#     Construct the Instances in detectron2/strutures/instances.py in a very simple way.
#     """

#     def __init__(self, image_shape):
#         self.image_shape = image_shape
#         self.gt_classes = None
#         self.gt_masks = None


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
                 to_mask=True):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge
        self.to_mask = to_mask

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
        # image_path, label_path = '/ssd2/tangshiyu/PaddleSeg/data/ADEChallengeData2016/images/training/ADE_train_00019747.jpg', \
        #                 '/ssd2/tangshiyu/PaddleSeg/data/ADEChallengeData2016/annotations/training/ADE_train_00019747.png'
        data['img'] = image_path
        data['gt_fields'] = [
        ]  # If key in gt_fields, the data[key] have transforms synchronous.

        if self.mode == 'val':
            data = self.transforms(data)
            label = np.asarray(Image.open(label_path))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            # label = label - 1
            label = label[np.newaxis, :, :]
            data['label'] = label
        else:
            data['label'] = label_path
            data['gt_fields'].append('label')
            data = self.transforms(data)
            # data['label'] = data['label'] - 1
            # Recover the ignore pixels adding by transform
            # data['label'][data['label'] == 254] = 255
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask

        if self.to_mask:
            # ignore the pad image with size_divisibility here
            sem_seg_gt = data['label']
            instances = {"image_shape": data['img'].shape}
            # instances = Instances(data['img'].shape)
            classes = np.unique(sem_seg_gt)  # 255
            classes = classes[classes != self.ignore_index]

            # compat with dataloader
            classes_cpt = np.array([
                self.ignore_index
                for i in range(self.num_classes - len(classes))
            ])
            classes_cpt = np.append(classes, classes_cpt)
            instances["gt_classes"] = paddle.to_tensor(
                classes_cpt, dtype="int64")  # all 255
            # instances["gt_classes"] = paddle.to_tensor([255 for i in range(self.num_classes)], dtype='int64')

            masks = []
            for cid in classes:
                masks.append(sem_seg_gt == cid)  # [C, H, W] 

            shape = [self.num_classes - len(masks)] + list(data['label'].shape)
            masks_cpt = paddle.zeros(shape=shape)

            # ignore bitmask in masks, let's see what we are missing here?
            # stack does not have kernel for {data_type[bool];
            # stack empyty masks
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances['gt_masks'] = paddle.zeros(
                    (150, sem_seg_gt.shape[-2],
                     sem_seg_gt.shape[-1]))  #150, 512, 512
                # print("image_path", image_path, classes)

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
        # print('data[\'instances\'][\'gt_masks\'].shape',  data['instances']['gt_masks'].shape, sem_seg_gt.shape, sem_seg_gt.mean(), len(masks), data['img'].shape)

        # batch data con only contains: tensor, numpy.ndarray, dict, list, number
        # ValueError: (InvalidArgument) Dims of all Inputs(X) must be the same
        return data


if __name__ == "__main__":
    d = ADE20K([], '/ssd2/tangshiyu/PaddleSeg/data/ADEChallengeData2016/')
    for i in range(len(d)):
        # print(data)
        data = d[i]
