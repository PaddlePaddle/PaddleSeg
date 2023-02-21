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

    def update_dataset(self, dataset_dir, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'Dataset'
        if dataset_type == 'Dataset':
            ds_cfg = self._make_custom_dataset_config(dataset_dir)
            # For custom datasets, we do not reset the existing dataset configs.
            # Note however that the user may have to manually set `num_classes`, etc.
            self.update(ds_cfg)
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

    def _make_custom_dataset_config(self, dataset_root_path):
        # TODO: Description of dataset protocol
        return {
            'train_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'train_path': os.path.join(dataset_root_path, 'train.txt'),
                'mode': 'train'
            },
            'val_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'val_path': os.path.join(dataset_root_path, 'val.txt'),
                'mode': 'val'
            },
        }

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
