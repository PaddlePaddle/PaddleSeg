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

import yaml
import paddle
import paddleseg
from paddleseg.utils import logger

from . import manager


class Config(paddleseg.cvlibs.Config):
    """
    A configuration class dedicated to panoptic segmentation tasks.

    This class inherits from `paddleseg.cvlibs.Config`. We add a few new functions
        to accomodate the panseg task.
    """

    @property
    def optimizer_config(self):
        # Overwrite `optimizer_config()` instead of `optimizer()` for simplicity
        args = super().optimizer_config
        assert 'grad_clip' not in args
        if 'grad_clip' in self.dic:
            args['grad_clip'] = self.grad_clip
        return args

    @property
    def grad_clip(self):
        if 'grad_clip' not in self.dic:
            raise RuntimeError(
                'No `grad_clip` specified in the configuration file.')
        params = self.dic.get('grad_clip')
        grad_clip_type = params.pop('type')
        if grad_clip_type not in ('ClipGradByGlobalNorm', 'ClipGradByNorm',
                                  'ClipGradByValue'):
            raise ValueError(
                f"{grad_clip_type} is not a supported gradient clipping strategy."
            )
        grad_clip = getattr(paddle.nn, grad_clip_type)(**params)
        return grad_clip

    @property
    def postprocessor(self):
        def _set_attr_if_not_exists(pp_cfg, attr_name, default=None):
            if attr_name not in pp_cfg:
                attr_val = getattr(self.val_dataset, attr_name, default)
                logger.info(
                    f"Will use `{attr_name}={attr_val}` for postprocessing.")
                pp_cfg[attr_name] = attr_val

        pp_cfg = self.dic.get('postprocessor').copy()
        if not hasattr(self, '_postprocessor'):
            _set_attr_if_not_exists(pp_cfg, 'num_classes', 1)
            _set_attr_if_not_exists(pp_cfg, 'thing_ids', [])
            _set_attr_if_not_exists(pp_cfg, 'label_divisor', 1000)
            _set_attr_if_not_exists(pp_cfg, 'ignore_index', 255)
            # XXX: Since PaddleSeg does not provide API to plug-in new types of 
            # components, we have to manually create the postprocessor object here.
            self._postprocessor = create_object(pp_cfg,
                                                [manager.POSTPROCESSORS])
        return self._postprocessor

    def __str__(self):
        # Disable anchors and aliases
        class _NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        return yaml.dump(self.dic, Dumper=_NoAliasDumper)


def load_component_class(com_name, com_list=None):
    """
    This function is modified based on
    https://github.com/PaddlePaddle/PaddleSeg/blob/7e076fa14864b5c01e1381a850e3d3863ee85581/paddleseg/cvlibs/config.py

    Compared to the original version, this function supports using customized `com_list`.
    """
    if com_list is None:
        com_list = [
            manager.MODELS, manager.BACKBONES, manager.DATASETS,
            manager.TRANSFORMS, manager.LOSSES, manager.POSTPROCESSORS
        ]

    for com in com_list:
        if com_name in com.components_dict:
            return com[com_name]

    raise RuntimeError("The specified component ({}) was not found.".format(
        com_name))


def create_object(cfg, com_list=None):
    """
    This function is modified based on
    https://github.com/PaddlePaddle/PaddleSeg/blob/7e076fa14864b5c01e1381a850e3d3863ee85581/paddleseg/cvlibs/config.py

    Compared to the original version, this function supports using customized `com_list`.
    """
    cfg = cfg.copy()
    if 'type' not in cfg:
        raise RuntimeError("No object information in {}.".format(cfg))

    is_meta_type = lambda item: isinstance(item, dict) and 'type' in item
    component = load_component_class(cfg.pop('type'), com_list)

    params = {}
    for key, val in cfg.items():
        if is_meta_type(val):
            params[key] = create_object(val, com_list)
        elif isinstance(val, list):
            params[key] = [
                create_object(item, com_list) if is_meta_type(item) else item
                for item in val
            ]
        else:
            params[key] = val

    return component(**params)
