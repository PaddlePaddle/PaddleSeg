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

import copy
from typing import Any, Optional

import yaml
import paddle
from paddleseg.utils import logger
from paddleseg.cvlibs import Builder
from paddleseg.utils.utils import CachedProperty as cached_property

from cvlibs import Config
from batch_transforms import BoxMaskGenerator
from . import manager
from utils import utils as cps_utils


class CPSBuilder(Builder):
    """
    This class is responsible for building components for semantic segmentation. 
    """

    def __init__(self, config, comp_list=None):
        if comp_list is None:
            comp_list = [
                manager.MODELS, manager.BACKBONES, manager.DATASETS,
                manager.TRANSFORMS, manager.LOSSES, manager.OPTIMIZERS,
                manager.BATCH_TRANSFORMS
            ]
        super().__init__(config, comp_list)

    @cached_property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.config.model_cfg
        assert model_cfg != {}, \
            'No model specified in the configuration file.'

        if self.config.train_dataset_cfg['type'] != 'Dataset':
            # check and synchronize the num_classes in model config and dataset class
            assert hasattr(self.train_dataset_class, 'NUM_CLASSES'), \
                'If train_dataset class is not `Dataset`, it must have `NUM_CLASSES` attr.'
            num_classes = getattr(self.train_dataset_class, 'NUM_CLASSES')
            if 'num_classes' in model_cfg:
                assert model_cfg['num_classes'] == num_classes, \
                    'The num_classes is not consistent for model config ({}) ' \
                    'and train_dataset class ({}) '.format(model_cfg['num_classes'], num_classes)
            else:
                logger.warning(
                    'Add the `num_classes` in train_dataset class to '
                    'model config. We suggest you manually set `num_classes` in model config.'
                )
                model_cfg['num_classes'] = num_classes
            # check and synchronize the in_channels in model config and dataset class
            assert hasattr(self.train_dataset_class, 'IMG_CHANNELS'), \
                'If train_dataset class is not `Dataset`, it must have `IMG_CHANNELS` attr.'
            in_channels = getattr(self.train_dataset_class, 'IMG_CHANNELS')
            x = cps_utils.get_in_channels(model_cfg)
            if x is not None:
                assert x == in_channels, \
                    'The in_channels in model config ({}) and the img_channels in train_dataset ' \
                    'class ({}) is not consistent'.format(x, in_channels)
            else:
                model_cfg = cps_utils.set_in_channels(model_cfg, in_channels)
                logger.warning(
                    'Add the `in_channels` in train_dataset class to '
                    'model config. We suggest you manually set `in_channels` in model config.'
                )

        self.show_msg('model', model_cfg)
        return self.build_component(model_cfg)

    def _build_lr_scheduler(self, lr_cfg) -> paddle.optimizer.lr.LRScheduler:
        assert lr_cfg != {}, \
            'No lr_scheduler specified in the configuration file.'
        use_warmup = False
        if 'warmup_iters' in lr_cfg:
            use_warmup = True
            warmup_iters = lr_cfg.pop('warmup_iters')
            assert 'warmup_start_lr' in lr_cfg, \
                "When use warmup, please set warmup_start_lr and warmup_iters in lr_scheduler"
            warmup_start_lr = lr_cfg.pop('warmup_start_lr')
            end_lr = lr_cfg['learning_rate']

        # calculate iters
        total_imgs = len(self.train_dataset) + len(
            self.unsupervised_train_dataset)
        num_train_imgs = total_imgs // self.config.labeled_ratio
        num_unsup_imgs = total_imgs - num_train_imgs
        max_samples = max(num_train_imgs, num_unsup_imgs)
        niters_per_epoch = max_samples // self.config.batch_size
        iters = niters_per_epoch * self.config.nepochs
        lr_type = lr_cfg.pop('type')
        if lr_type == 'PolynomialDecay':
            iters = iters - warmup_iters if use_warmup else iters
            iters = max(iters, 1)
            lr_cfg.setdefault('decay_steps', iters)

        try:
            lr_sche = getattr(paddle.optimizer.lr, lr_type)(**lr_cfg)
        except Exception as e:
            raise RuntimeError(
                "Create {} has failed. Please check lr_scheduler in config. "
                "The error message: {}".format(lr_type, e))

        if use_warmup:
            lr_sche = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_iters,
                start_lr=warmup_start_lr,
                end_lr=end_lr)

        return lr_sche

    def _build_optimizer(self, opt_cfg) -> paddle.optimizer.Optimizer:
        assert opt_cfg != {}, \
            'No optimizer specified in the configuration file.'
        # For compatibility
        if opt_cfg['type'] == 'adam':
            opt_cfg['type'] = 'Adam'
        if opt_cfg['type'] == 'sgd':
            opt_cfg['type'] = 'SGD'
        if opt_cfg['type'] == 'SGD' and 'momentum' in opt_cfg:
            opt_cfg['type'] = 'Momentum'
            logger.info('If the type is SGD and momentum in optimizer config, '
                        'the type is changed to Momentum.')
        self.show_msg('optimizer', opt_cfg)
        opt = self.build_component(opt_cfg)
        return opt

    @cached_property
    def optimizer_l(self) -> paddle.optimizer.Optimizer:
        opt_cfg = self.config.optimizer_l_cfg
        lr_cfg = self.config.lr_scheduler_l_cfg

        opt = self._build_optimizer(opt_cfg)
        lr = self._build_lr_scheduler(lr_cfg)

        opt = opt(self.model.branch1, lr)
        return opt

    @cached_property
    def optimizer_r(self) -> paddle.optimizer.Optimizer:
        opt_cfg = self.config.optimizer_r_cfg
        lr_cfg = self.config.lr_scheduler_r_cfg

        opt = self._build_optimizer(opt_cfg)
        lr = self._build_lr_scheduler(lr_cfg)

        opt = opt(self.model.branch2, lr)
        return opt

    @cached_property
    def loss(self) -> dict:
        loss_cfg = self.config.loss_cfg
        assert loss_cfg != {}, \
            'No loss specified in the configuration file.'
        return self._build_loss('loss', loss_cfg)

    def _build_loss(self, loss_name, loss_cfg: dict):
        def _check_helper(loss_cfg, ignore_index):
            if 'ignore_index' not in loss_cfg:
                loss_cfg['ignore_index'] = ignore_index
                logger.warning('Add the `ignore_index` in train_dataset ' \
                    'class to {} config. We suggest you manually set ' \
                    '`ignore_index` in {} config.'.format(loss_name, loss_name)
                )
            else:
                assert loss_cfg['ignore_index'] == ignore_index, \
                    'the ignore_index in loss and train_dataset must be the same. Currently, loss ignore_index = {}, '\
                    'train_dataset ignore_index = {}'.format(loss_cfg['ignore_index'], ignore_index)

        # check and synchronize the ignore_index in model config and dataset class
        if self.config.train_dataset_cfg['type'] != 'Dataset':
            assert hasattr(self.train_dataset_class, 'IGNORE_INDEX'), \
                'If train_dataset class is not `Dataset`, it must have `IGNORE_INDEX` attr.'
            ignore_index = getattr(self.train_dataset_class, 'IGNORE_INDEX')
            for loss_cfg_i in loss_cfg['types']:
                if loss_cfg_i['type'] == 'MixedLoss':
                    for loss_cfg_j in loss_cfg_i['losses']:
                        _check_helper(loss_cfg_j, ignore_index)
                else:
                    _check_helper(loss_cfg_i, ignore_index)

        self.show_msg(loss_name, loss_cfg)
        loss_dict = {'coef': loss_cfg['coef'], "types": []}
        for item in loss_cfg['types']:
            loss_dict['types'].append(self.build_component(item))
        return loss_dict

    @cached_property
    def train_dataset(self) -> paddle.io.Dataset:
        dataset_cfg = self.config.train_dataset_cfg
        assert dataset_cfg != {}, \
            'No train_dataset specified in the configuration file.'
        self.show_msg('train_dataset', dataset_cfg)
        dataset = self.build_component(dataset_cfg)
        assert len(dataset) != 0, \
            'The number of samples in train_dataset is 0. Please check whether the dataset is valid.'
        return dataset

    @cached_property
    def unsupervised_train_dataset(self) -> paddle.io.Dataset:
        dataset_cfg = self.config.unsupervised_train_dataset_cfg
        assert dataset_cfg != {}, \
            'No unsupervised train_dataset specified in the configuration file.'
        self.show_msg('train_dataset', dataset_cfg)
        dataset = self.build_component(dataset_cfg)
        assert len(dataset) != 0, \
            'The number of samples in unsupervised train_dataset is 0. Please check whether the dataset is valid.'
        return dataset

    @cached_property
    def val_dataset(self) -> paddle.io.Dataset:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        self.show_msg('val_dataset', dataset_cfg)
        dataset = self.build_component(dataset_cfg)
        if len(dataset) == 0:
            logger.warning(
                'The number of samples in val_dataset is 0. Please ensure this is the desired behavior.'
            )
        return dataset

    @cached_property
    def train_dataset_class(self) -> Any:
        dataset_cfg = self.config.train_dataset_cfg
        assert dataset_cfg != {}, \
            'No train_dataset specified in the configuration file.'
        dataset_type = dataset_cfg.get('type')
        return self.load_component_class(dataset_type)

    @cached_property
    def unsupervised_train_dataset_class(self) -> Any:
        dataset_cfg = self.config.unsupervised_train_dataset_cfg
        assert dataset_cfg != {}, \
            'No unsupervised train_dataset specified in the configuration file.'
        dataset_type = dataset_cfg.get('type')
        return self.load_component_class(dataset_type)

    @cached_property
    def val_dataset_class(self) -> Any:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        dataset_type = dataset_cfg.get('type')
        return self.load_component_class(dataset_type)

    @cached_property
    def val_transforms(self) -> list:
        dataset_cfg = self.config.val_dataset_cfg
        assert dataset_cfg != {}, \
            'No val_dataset specified in the configuration file.'
        transforms = []
        for item in dataset_cfg.get('transforms', []):
            transforms.append(self.build_component(item))
        return transforms

    @cached_property
    def batch_transforms(self) -> BoxMaskGenerator:
        batch_transforms_cfg = self.config.batch_transforms_cfg
        assert batch_transforms_cfg != {}, \
            'No batch_transform_cfg specified in the configuration file.'
        self.show_msg('batch_transforms', batch_transforms_cfg)
        batch_transforms = self.build_component(batch_transforms_cfg)
        return batch_transforms
