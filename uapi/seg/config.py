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
            ds_cfg = self._make_custom_dataset_config(dataset_path)
            # For custom datasets, we do not reset the existing dataset configs.
            # Note however that the user may have to manually set `num_classes`, etc.
            self.update(ds_cfg)
        else:
            raise ValueError(f"{dataset_type} is not supported.")

    def update_optimizer(self, optimizer_type):
        _SUPPORTED_TYPES = ['SGD', 'Momentum', 'Adam', 'AdamW']
        if optimizer_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"{optimizer_type} is not supported. Suppported types: {_SUPPORTED_TYPES}"
            )
        else:
            # Overwrite optimizer config
            self.set_val('optimizer', {'type': optimizer_type})

    def update_backbone(self, backbone_type):
        _SUPPORTED_BACKBONE_INFO = {
            'GhostNet_x1_0': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/ghostnet_x1_0.zip'
            },
            'HRNet_W18': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/hrnet_w18_ssld.tar.gz'
            },
            'HRNet_W48': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/hrnet_w48_ssld.tar.gz'
            },
            'Lite_HRNet_18': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/lite_hrnet_18.tar.gz'
            },
            'MixVisionTransformer_B0': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b0.tar.gz'
            },
            'MixVisionTransformer_B1': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b1.tar.gz'
            },
            'MixVisionTransformer_B2': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b2.tar.gz'
            },
            'MixVisionTransformer_B3': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b3.tar.gz'
            },
            'MixVisionTransformer_B4': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b4.tar.gz'
            },
            'MixVisionTransformer_B5': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/mix_vision_transformer_b5.tar.gz'
            },
            'MobileNetV2_x1_0': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/mobilenetv2_x1_0_ssld.tar.gz'
            },
            'MobileNetV3_large_x1_0': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/mobilenetv3_large_x1_0_ssld.tar.gz'
            },
            'MobileNetV3_large_x1_0_os8': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/mobilenetv3_large_x1_0_ssld.tar.gz'
            },
            'ResNet101_vd': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz'
            },
            'ResNet50_vd': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz'
            },
            'ShuffleNetV2_x1_0': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/shufflenetv2_x1_0.zip'
            },
            'TopTransformer_Base': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/topformer_base_imagenet_pretrained.zip'
            },
            'TopTransformer_Small': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/topformer_small_imagenet_pretrained.zip'
            },
            'TopTransformer_Tiny': {
                'pretrained':
                'https://paddleseg.bj.bcebos.com/dygraph/backbone/topformer_tiny_imagenet_pretrained.zip'
            },
            'UHRNet_W18_Small': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/uhrnetw18_small_imagenet.tar.gz'
            },
            'UHRNet_W48': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/backbone/uhrnetw48_imagenet.tar.gz'
            },
            'VisionTransformer': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/pretrained_models/vit_base_patch16_384_augreg.tar.gz'
            },
            'ViT_large_patch16_384': {
                'pretrained':
                'https://bj.bcebos.com/paddleseg/dygraph/vit_large_patch16_384.tar.gz'
            },
        }
        _SUPPORTED_TYPES = sorted(_SUPPORTED_BACKBONE_INFO.keys())
        if backbone_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"{backbone_type} is not supported. Suppported types: {_SUPPORTED_TYPES}"
            )
        else:
            if 'model' not in self:
                raise RuntimeError(
                    "Not able to update backbone config, because no model config was found."
                )
            elif 'backbone' not in self.model:
                raise RuntimeError(
                    "This model does not support changing the backbone.")
            backbone_info = _SUPPORTED_BACKBONE_INFO[backbone_type]
            # Overwrite backbone config
            backbone_cfg = {'type': backbone_type, ** backbone_info}
            self.model['backbone'] = backbone_cfg

    def update_lr_scheduler(self, lr_scheduler_type):
        if lr_scheduler_type == 'PolynomialDecay':
            self.set_val('lr_scheduler', {
                'type': 'PolynomialDecay',
                'learning_rate': 0.01,
                'end_lr': 0,
                'power': 0.9
            })
        else:
            raise ValueError(f"{lr_scheduler_type} is not supported.")

    def update_batch_size(self, batch_size, mode='train'):
        if mode == 'train':
            self.set_val('batch_size', batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in '{mode}' mode is not supported.")

    def update_weight_decay(self, weight_decay):
        if 'optimizer' not in self:
            raise RuntimeError(
                "Not able to update weight decay, because no optimizer config was found."
            )
        self.optimizer['weight_decay'] = weight_decay

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
