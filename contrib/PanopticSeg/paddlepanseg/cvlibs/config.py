# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddleseg
from paddleseg.cvlibs import builder
from paddleseg.utils import logger
from paddleseg.utils.utils import CachedProperty as cached_property

from . import manager


class Config(paddleseg.cvlibs.Config):
    """
    A configuration class dedicated to panoptic segmentation tasks.

    This class inherits from `paddleseg.cvlibs.Config`. We add a few new functions
        to accomodate the panseg task.
    """

    @property
    def postprocessor_cfg(self):
        return self.dic.get('postprocessor', {}).copy()

    @property
    def export_cfg(self):
        return self.dic.get('export', {}).copy()

    @property
    def runner_cfg(self):
        return self.dic.get('runner', {}).copy()


class PanSegBuilder(builder.SegBuilder):
    def __init__(self, config, comp_list=None):
        if comp_list is None:
            comp_list = [
                manager.MODELS, manager.BACKBONES, manager.DATASETS,
                manager.TRANSFORMS, manager.LOSSES, manager.OPTIMIZERS,
                manager.POSTPROCESSORS, manager.RUNNERS
            ]
        super().__init__(config, comp_list)

    @cached_property
    def postprocessor(self):
        def _set_attr_if_not_exists(pp_cfg, attr_name, default=None):
            if attr_name not in pp_cfg:
                attr_val = getattr(self.val_dataset, attr_name, default)
                logger.info(
                    f"Will use `{attr_name}={attr_val}` for postprocessing.")
                pp_cfg[attr_name] = attr_val

        pp_cfg = self.config.postprocessor_cfg
        if pp_cfg == {}:
            raise RuntimeError(
                "No `postprocessor` is specified in the configuration file.")
        _set_attr_if_not_exists(pp_cfg, 'num_classes', 1)
        _set_attr_if_not_exists(pp_cfg, 'thing_ids', [])
        _set_attr_if_not_exists(pp_cfg, 'label_divisor', 1000)
        _set_attr_if_not_exists(pp_cfg, 'ignore_index', 255)
        postprocessor = self.build_component(pp_cfg)
        return postprocessor

    @cached_property
    def runner(self):
        runner_cfg = self.config.runner_cfg
        if runner_cfg == {}:
            raise RuntimeError(
                "No `runner` is specified in the configuration file.")
        runner = self.build_component(runner_cfg)
        return runner


def make_default_builder(*args, **kwargs):
    return PanSegBuilder(*args, **kwargs)
