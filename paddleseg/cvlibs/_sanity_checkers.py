# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class SanityChecker(object):
    def __init__(self, rule_list, allow_update=True):
        super().__init__()
        self.rule_list = rule_list
        self.allow_update = allow_update

    def apply_rule(self, k, cfg):
        rule = self.rule_list[k]
        try:
            rule.apply(cfg, self.allow_update)
        except Exception as e:
            raise RuntimeError(
                "Sanity check failed. There should be some problems with your config file. "
                "Please check it carefully.\n"
                f"The failed rule is {rule.__class__.__name__}, and the error message is: \n{str(e)}"
            )

    def apply_all_rules(self, cfg):
        for i, rule in enumerate(self.rule_list):
            self.apply_rule(i, cfg)
            # Do nothing here as `self.apply_rule()` already handles the exceptions


class _Rule(object):
    def check_and_correct(self, cfg):
        # Be free to add in-place modification here
        raise NotImplementedError

    def apply(self, cfg, allow_update):
        if not allow_update:
            # If update is not allowed, make a deep copy, such that the original 
            # `cfg` will remain unchanged.
            cfg = copy.deepcopy(cfg)
        self.check_and_correct(cfg)


class DefaultPrimaryRule(_Rule):
    def check_and_correct(self, cfg):
        assert cfg.dic.get('model', None) is not None, \
            'No model specified in the configuration file.'
        assert cfg.train_dataset_config or self.val_dataset_config, \
            'One of `train_dataset` or `val_dataset should be given, but there are none.'


class DefaultLossRule(_Rule):
    def __init__(self, loss_name):
        super().__init__()
        self.loss_name = loss_name

    def check_and_correct(self, cfg):
        loss_cfg = cfg.dic.get(self.loss_name, None)
        if loss_cfg is None:
            return

        assert 'types' in loss_cfg and 'coef' in loss_cfg, \
                'Loss config should contain keys of "types" and "coef"'
        len_types = len(loss_cfg['types'])
        len_coef = len(loss_cfg['coef'])
        if len_types != len_coef:
            if len_types == 1:
                loss_cfg['types'] = loss_cfg['types'] * len_coef
            else:
                raise ValueError(
                    "For loss config, the length of types should be 1 "
                    "or be equal to coef , but they are {} and {}.".format(
                        len_types, len_coef))


class DefaultSyncNumClassesRule(_Rule):
    def check_and_correct(self, cfg):
        num_classes_set = set()

        if cfg.dic['model'].get('num_classes', None) is not None:
            num_classes_set.add(cfg.dic['model'].get('num_classes'))
        if cfg.train_dataset_config:
            if hasattr(cfg.train_dataset_class, 'NUM_CLASSES'):
                num_classes_set.add(cfg.train_dataset_class.NUM_CLASSES)
            if 'num_classes' in cfg.train_dataset_config:
                num_classes_set.add(cfg.train_dataset_config['num_classes'])
        if cfg.val_dataset_config:
            if hasattr(cfg.val_dataset_class, 'NUM_CLASSES'):
                num_classes_set.add(cfg.val_dataset_class.NUM_CLASSES)
            if 'num_classes' in cfg.val_dataset_config:
                num_classes_set.add(cfg.val_dataset_config['num_classes'])

        if len(num_classes_set) == 0:
            raise ValueError(
                '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
            )
        elif len(num_classes_set) > 1:
            raise ValueError(
                '`num_classes` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(num_classes_set))

        num_classes = num_classes_set.pop()
        cfg.dic['model']['num_classes'] = num_classes
        if cfg.train_dataset_config and \
            (not hasattr(cfg.train_dataset_class, 'NUM_CLASSES')):
            cfg.dic['train_dataset']['num_classes'] = num_classes
        if cfg.val_dataset_config and \
            (not hasattr(cfg.val_dataset_class, 'NUM_CLASSES')):
            cfg.dic['val_dataset']['num_classes'] = num_classes


class DefaultSyncImgChannelsRule(_Rule):
    def check_and_correct(self, cfg):
        img_channels_set = set()
        model_cfg = cfg.dic['model']

        # If the model has backbone, in_channels is the input params of backbone.
        # Otherwise, in_channels is the input params of the model.
        if 'backbone' in model_cfg:
            x = model_cfg['backbone'].get('in_channels', None)
            if x is not None:
                img_channels_set.add(x)
        elif model_cfg.get('in_channels', None) is not None:
            img_channels_set.add(model_cfg.get('in_channels'))
        if cfg.train_dataset_config and \
            ('img_channels' in cfg.train_dataset_config):
            img_channels_set.add(cfg.train_dataset_config['img_channels'])
        if cfg.val_dataset_config and \
            ('img_channels' in cfg.val_dataset_config):
            img_channels_set.add(cfg.val_dataset_config['img_channels'])

        if len(img_channels_set) > 1:
            raise ValueError(
                '`img_channels` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(img_channels_set))

        img_channels = 3 if len(img_channels_set) == 0 \
            else img_channels_set.pop()
        if 'backbone' in model_cfg:
            cfg.dic['model']['backbone']['in_channels'] = img_channels
        else:
            cfg.dic['model']['in_channels'] = img_channels
        if cfg.train_dataset_config and \
            cfg.train_dataset_config['type'] == "Dataset":
            cfg.dic['train_dataset']['img_channels'] = img_channels
        if cfg.val_dataset_config and \
            cfg.val_dataset_config['type'] == "Dataset":
            cfg.dic['val_dataset']['img_channels'] = img_channels


class DefaultSyncIgnoreIndexRule(_Rule):
    def __init__(self, loss_name):
        super().__init__()
        self.loss_name = loss_name

    def check_and_correct(self, cfg):
        def _check_ignore_index(loss_cfg, dataset_ignore_index):
            if 'ignore_index' in loss_cfg:
                assert loss_cfg['ignore_index'] == dataset_ignore_index, \
                    'the ignore_index in loss and train_dataset must be the same. Currently, loss ignore_index = {}, '\
                    'train_dataset ignore_index = {}'.format(loss_cfg['ignore_index'], dataset_ignore_index)

        loss_cfg = cfg.dic.get(self.loss_name, None)
        if loss_cfg is None:
            return

        dataset_ignore_index = cfg.train_dataset.ignore_index
        for loss_cfg_i in loss_cfg['types']:
            if loss_cfg_i['type'] == 'MixedLoss':
                for loss_cfg_j in loss_cfg_i['losses']:
                    _check_ignore_index(loss_cfg_j, dataset_ignore_index)
                    loss_cfg_j['ignore_index'] = dataset_ignore_index
            else:
                _check_ignore_index(loss_cfg_i, dataset_ignore_index)
                loss_cfg_i['ignore_index'] = dataset_ignore_index
