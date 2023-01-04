# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddleseg
from paddleseg.cvlibs import builder
from paddleseg.cvlibs import config_checker as checker
from paddleseg.utils import logger
from paddleseg.utils.utils import CachedProperty as cached_property

from . import manager


class Config(paddleseg.cvlibs.Config):
    """
    A configuration class dedicated to panoptic segmentation tasks.

    This class inherits from `paddleseg.cvlibs.Config`. We add a few new functions
        to accomodate the panseg task.
    """

    @cached_property
    def optimizer_config(self):
        # Overwrite `optimizer_config()` instead of `optimizer()` for simplicity
        args = super().optimizer_config
        assert 'grad_clip' not in args
        if 'grad_clip' in self.dic:
            args['grad_clip'] = self.grad_clip
        return args

    @cached_property
    def grad_clip(self):
        if 'grad_clip' not in self.dic:
            raise RuntimeError(
                'No `grad_clip` is specified in the configuration file.')
        params = self.dic.get('grad_clip')
        grad_clip_type = params.pop('type')
        if grad_clip_type not in ('ClipGradByGlobalNorm', 'ClipGradByNorm',
                                  'ClipGradByValue'):
            raise ValueError(
                f"{grad_clip_type} is not a supported gradient clipping strategy."
            )
        grad_clip = getattr(paddle.nn, grad_clip_type)(**params)
        return grad_clip

    @cached_property
    def postprocessor(self):
        pp_cfg = self.dic.get('postprocessor').copy()
        self._postprocessor = self.builder.create_object(pp_cfg)
        return self._postprocessor

    @classmethod
    def _build_default_checker(cls):
        checker = super()._build_default_checker()
        checker.add_rule(DefaultSyncPostprocessorRule())
        return checker

    @classmethod
    def _build_default_component_builder(cls):
        com_list = [
            manager.MODELS, manager.BACKBONES, manager.DATASETS,
            manager.TRANSFORMS, manager.LOSSES, manager.POSTPROCESSORS
        ]
        component_builder = builder.DefaultComponentBuilder(com_list=com_list)
        return component_builder


class DefaultSyncPostprocessorRule(checker.Rule):
    def check_and_correct(self, cfg):
        def _set_attr_if_not_exists(pp_cfg, attr_name, default=None):
            if attr_name not in pp_cfg:
                attr_val = getattr(cfg.val_dataset, attr_name, default)
                logger.info(
                    f"Will use `{attr_name}={attr_val}` for postprocessing.")
                pp_cfg[attr_name] = attr_val

        if 'postprocessor' not in cfg.dic:
            raise RuntimeError("No `postprocessor` is specified in the configuration file.")
        pp_cfg = cfg.dic['postprocessor']
        _set_attr_if_not_exists(pp_cfg, 'num_classes', 1)
        _set_attr_if_not_exists(pp_cfg, 'thing_ids', [])
        _set_attr_if_not_exists(pp_cfg, 'label_divisor', 1000)
        _set_attr_if_not_exists(pp_cfg, 'ignore_index', 255)
