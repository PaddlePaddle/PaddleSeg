# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import glob

import numpy as np
import paddle
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import PIL.Image as Image

from transforms import PanopticTargetGenerator, SemanticTargetGenerator, InstanceTargetGenerator, RawPanopticTargetGenerator


@manager.DATASETS.add_component
class CityscapesPanoptic(paddle.io.Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        cityscapes/
        |--gtFine/
        |  |--train/
        |  |  |--aachen/
        |  |  |  |--*_color.png, *_instanceIds.png, *_labelIds.png, *_polygons.json,
        |  |  |  |--*_labelTrainIds.png
        |  |  |  |--...
        |  |--val/
        |  |--test/
        |  |--cityscapes_panoptic_train_trainId.json
        |  |--cityscapes_panoptic_train_trainId/
        |  |  |-- *_panoptic.png
        |  |--cityscapes_panoptic_val_trainId.json
        |  |--cityscapes_panoptic_val_trainId/
        |  |  |--  *_panoptic.png
        |--leftImg8bit/
        |  |--train/
        |  |--val/
        |  |--test/

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val'). Default: 'train'.
        ignore_stuff_in_offset (bool, optional): Whether to ignore stuff region when training the offset branch. Default: False.
        small_instance_area (int, optional): Instance which area less than given value is considered small. Default: 0.
        small_instance_weight (int, optional): The loss weight for small instance. Default: 1.
        stuff_area (int, optional): An Integer, remove stuff whose area is less tan stuff_area. Default: 2048.
    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 mode='train',
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 stuff_area=2048):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.ins_list = []
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.stuff_area = stuff_area

        if mode not in ['train', 'val']:
            raise ValueError(
                "mode should be 'train' or 'val' , but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'leftImg8bit')
        label_dir = os.path.join(self.dataset_root, 'gtFine')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )
        json_filename = os.path.join(
            self.dataset_root, 'gtFine',
            'cityscapes_panoptic_{}_trainId.json'.format(mode))
        dataset = json.load(open(json_filename))
        img_files = []
        label_files = []
        for img in dataset['images']:
            img_file_name = img['file_name']
            img_files.append(
                os.path.join(self.dataset_root, 'leftImg8bit', mode,
                             img_file_name.split('_')[0],
                             img_file_name.replace('_gtFine', '')))
        for ann in dataset['annotations']:
            ann_file_name = ann['file_name']
            label_files.append(
                os.path.join(self.dataset_root, 'gtFine',
                             'cityscapes_panoptic_{}_trainId'.format(
                                 mode), ann_file_name))
            self.ins_list.append(ann['segments_info'])

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]

        self.target_transform = PanopticTargetGenerator(
            self.ignore_index,
            self.rgb2id,
            self.thing_list,
            sigma=8,
            ignore_stuff_in_offset=ignore_stuff_in_offset,
            small_instance_area=small_instance_area,
            small_instance_weight=small_instance_weight)

        self.raw_semantic_generator = SemanticTargetGenerator(
            ignore_index=self.ignore_index, rgb2id=self.rgb2id)
        self.raw_instance_generator = InstanceTargetGenerator(self.rgb2id)
        self.raw_panoptic_generator = RawPanopticTargetGenerator(
            ignore_index=self.ignore_index,
            rgb2id=self.rgb2id,
            label_divisor=self.label_divisor)

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.

        Args:
            color: Ndarray or a tuple, color encoded image.

        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :,
                                                1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        dataset_dict = {}
        im, label = self.transforms(im=image_path, label=label_path)
        label_dict = self.target_transform(label, self.ins_list[idx])
        for key in label_dict.keys():
            dataset_dict[key] = label_dict[key]
        dataset_dict['image'] = im
        if self.mode == 'val':
            raw_label = np.asarray(Image.open(label_path))
            dataset_dict['raw_semantic_label'] = self.raw_semantic_generator(
                raw_label, self.ins_list[idx])['semantic']
            dataset_dict['raw_instance_label'] = self.raw_instance_generator(
                raw_label)['instance']
            dataset_dict['raw_panoptic_label'] = self.raw_panoptic_generator(
                raw_label, self.ins_list[idx])['panoptic']

        image = np.array(dataset_dict['image'])
        semantic = np.array(dataset_dict['semantic'])
        semantic_weights = np.array(dataset_dict['semantic_weights'])
        center = np.array(dataset_dict['center'])
        center_weights = np.array(dataset_dict['center_weights'])
        offset = np.array(dataset_dict['offset'])
        offset_weights = np.array(dataset_dict['offset_weights'])
        foreground = np.array(dataset_dict['foreground'])
        if self.mode == 'train':
            return image, semantic, semantic_weights, center, center_weights, offset, offset_weights, foreground
        elif self.mode == 'val':
            raw_semantic_label = np.array(dataset_dict['raw_semantic_label'])
            raw_instance_label = np.array(dataset_dict['raw_instance_label'])
            raw_panoptic_label = np.array(dataset_dict['raw_panoptic_label'])
            return image, raw_semantic_label, raw_instance_label, raw_panoptic_label
        else:
            raise ValueError(
                '{} is not surpported, please set it one of ("train", "val")'.
                format(self.mode))

    def __len__(self):
        return len(self.file_list)
