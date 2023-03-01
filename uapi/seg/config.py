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
from functools import lru_cache

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

    def update_dataset(self, dataset_dir, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'Dataset'
        if dataset_type == 'Dataset':
            ds_cfg = self._make_custom_dataset_config(dataset_dir)
            # For custom datasets, we reset the existing dataset configs.
            self.update(ds_cfg)
            if 'model' in self and 'num_classes' in self.model:
                num_classes = ds_cfg['train_dataset']['num_classes']
                self.model['num_classes'] = num_classes
        else:
            raise ValueError(f"{dataset_type} is not supported.")

    def update_learning_rate(self, learning_rate):
        if 'lr_scheduler' not in self:
            raise RuntimeError(
                "Not able to update learning rate, because no LR scheduler config was found."
            )
        self.lr_scheduler['learning_rate'] = learning_rate

    def update_batch_size(self, batch_size, mode='train'):
        if mode == 'train':
            self.set_val('batch_size', batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in '{mode}' mode is not supported.")

    def _get_epochs_iters(self):
        if 'iters' in self:
            return self.iters
        else:
            # Default iters
            return 1000

    def _get_learning_rate(self):
        if 'lr_scheduler' not in self or 'learning_rate' not in self.lr_scheduler:
            # Default lr
            return 0.0001
        else:
            return self.lr_scheduler['learning_rate']

    def _get_batch_size(self, mode='train'):
        if 'batch_size' in self:
            return self.batch_size
        else:
            # Default batch size
            return 4

    def _get_qat_epochs_iters(self):
        return self.get_epochs_iters()

    def _get_qat_learning_rate(self):
        return self.get_learning_rate()

    def _update_dy2st(self, dy2st):
        self.set_val('to_static_training', dy2st)

    def _make_custom_dataset_config(self, dataset_root_path):
        # TODO: Description of dataset protocol
        def _calc_avg_size(sizes):
            h_sum, w_sum = 0, 0
            for h, w in sizes.keys():
                h_sum += h
                w_sum += w
            return int(h_sum / len(sizes)), int(w_sum / len(sizes))

        def _round(num, mult=32):
            res = num + mult // 2
            res -= res % mult
            return res

        # Step 1: Extract meta info
        meta = self._extract_dataset_metadata(dataset_root_path, 'Dataset')

        # Step 2: Construct transforms
        if 'train.im_sizes' in meta:
            h, w = _calc_avg_size(meta['train.im_sizes'])
        else:
            # Default crop size
            h, w = 224, 224
        train_transforms = [{
            'type': 'RandomPaddingCrop',
            'crop_size': [_round(w), _round(h)]
        }, {
            'type': 'RandomHorizontalFlip'
        }, {
            'type': 'Normalize'
        }]
        if 'val.im_sizes' in meta:
            h, w = _calc_avg_size(meta['val.im_sizes'])
        else:
            # Default val target size
            h, w = 224, 224
        val_transforms = [{
            'type': 'Resize',
            'target_size': [_round(w), _round(h)]
        }, {
            'type': 'Normalize'
        }]

        # Step 3: Get number of classes
        # By default we set the number of classes to 2
        num_classes = meta.get('num_classes', 2)

        # Step 4: Construct dataset config
        return {
            'train_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'train_path': os.path.join(dataset_root_path, 'train.txt'),
                'mode': 'train',
                'num_classes': num_classes,
                'transforms': train_transforms
            },
            'val_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'val_path': os.path.join(dataset_root_path, 'val.txt'),
                'mode': 'val',
                'num_classes': num_classes,
                'transforms': val_transforms
            },
        }

    # TODO: A full scanning of dataset can be time-consuming. 
    # Maybe we should cache the result to disk to avoid rescanning in another run?
    @lru_cache(8)
    def _extract_dataset_metadata(self, dataset_root_path, dataset_type):
        from .check_dataset import check_dataset
        meta = check_dataset(dataset_root_path, dataset_type)
        if not meta:
            # Return an empty dict
            return dict()
        else:
            return meta

    def __repr__(self):
        # According to
        # https://github.com/PaddlePaddle/PaddleSeg/blob/e46a184777416d92e48f6341ee995c30aabb0930/paddleseg/utils/utils.py#L46
        dic = self.dict
        msg = "\n---------------Config Information---------------\n"
        ordered_module = ('batch_size', 'iters', 'train_dataset', 'val_dataset',
                          'optimizer', 'lr_scheduler', 'loss', 'model')
        all_module = set(dic.keys())
        for module in ordered_module:
            if module in dic:
                module_dic = {module: dic[module]}
                msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
                all_module.remove(module)
        for module in all_module:
            module_dic = {module: dic[module]}
            msg += str(yaml.dump(module_dic, Dumper=NoAliasDumper))
        msg += "------------------------------------------------\n"
        return msg
