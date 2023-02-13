# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import yaml
from paddleseg.utils import NoAliasDumper
from paddleseg.cvlibs.config import parse_from_yaml, merge_config_dicts

from ..base import BaseConfig


class SegConfig(BaseConfig):
    def update(self, dict_like_obj):
        dict_ = merge_config_dicts(dict_like_obj, self.dict)
        self.reset_from_dict(dict_)

    def load(self, config_path):
        dict_ = parse_from_yaml(config_path)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_path):
        with open(config_path, 'w') as f:
            yaml.dump(self.dict, f, Dumper=NoAliasDumper)

    def update_dataset(self, dataset_path, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'Dataset'
        if dataset_type == 'Dataset':
            ds_cfg = self._make_dataset_config(dataset_path)
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        self.update(ds_cfg)

    def update_optimizer(self, optimizer_type):
        # Not yet implemented
        raise NotImplementedError

    def update_backbone(self, backbone_type):
        # Not yet implemented
        raise NotImplementedError

    def update_lr_scheduler(self, lr_scheduler_type):
        # Not yet implemented
        raise NotImplementedError

    def update_batch_size(self, batch_size, mode='train'):
        # Not yet implemented
        raise NotImplementedError

    def update_weight_decay(self, weight_decay):
        # Not yet implemented
        raise NotImplementedError

    def _make_dataset_config(self, dataset_root_path):
        # TODO: Description of dataset protocol
        return {
            'train_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'train_path': os.path.join(dataset_root_path, 'train.txt'),
                'num_classes': 2,
                'transforms': [{
                    'type': 'Resize',
                    'target_size': [398, 224]
                }, {
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type': 'RandomDistort',
                    'brightness_range': 0.4,
                    'contrast_range': 0.4,
                    'saturation_range': 0.4
                }, {
                    'type': 'Normalize'
                }],
                'mode': 'train'
            },
            'val_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'val_path': os.path.join(dataset_root_path, 'val.txt'),
                'num_classes': 2,
                'transforms': [{
                    'type': 'Resize',
                    'target_size': [398, 224]
                }, {
                    'type': 'Normalize'
                }],
                'mode': 'val'
            },
        }
