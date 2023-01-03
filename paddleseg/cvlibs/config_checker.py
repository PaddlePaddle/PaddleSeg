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

import copy
from paddleseg.utils import logger


class ConfigChecker(object):
    """
    This class performs sanity checks on configuration objects and (optionally) updates the configurations
        (e.g., synchronize specific key-value pairs) based on a set of rules. 

    Args:
        rule_list (list): A list of rules on which all checks and updates are based.
        allow_update (bool, optional): Whether or not to allow updating the configuration object.
    """

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
                "Sanity check on the configuration file has failed. "
                "There should be some problems with your config file. "
                "Please check it carefully.\n"
                f"The failed rule is {rule.__class__.__name__}, and the error message is: \n{str(e)}"
            )

    def apply_all_rules(self, cfg):
        for i in range(len(self.rule_list)):
            self.apply_rule(i, cfg)

    def add_rule(self, rule):
        self.rule_list.append(rule)


class Rule(object):
    def check_and_correct(self, cfg):
        raise NotImplementedError

    def apply(self, cfg, allow_update):
        if not allow_update:
            cfg = copy.deepcopy(cfg)
        self.check_and_correct(cfg)


class DefaultPrimaryRule(Rule):
    def check_and_correct(self, cfg):
        items = [
            'batch_size', 'iters', 'train_dataset', 'val_dataset', 'optimizer',
            'lr_scheduler', 'loss', 'model'
        ]
        for i in items:
            assert i in cfg.dic, \
            'No {} specified in the configuration file.'.format(i)


class DefaultLossRule(Rule):
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


class DefaultSyncNumClassesRule(Rule):
    def check_and_correct(self, cfg):
        model_config = cfg.dic['model']
        train_dataset_config = cfg.dic['train_dataset']
        val_dataset_config = cfg.dic['val_dataset']
        value_set = set()
        value_name = 'num_classes'

        if value_name in model_config:
            value_set.add(model_config[value_name])
        if value_name in train_dataset_config:
            value_set.add(train_dataset_config[value_name])
        if value_name in val_dataset_config:
            value_set.add(val_dataset_config[value_name])
        if hasattr(cfg.train_dataset_class, 'NUM_CLASSES'):
            value_set.add(cfg.train_dataset_class.NUM_CLASSES)
        if hasattr(cfg.val_dataset_class, 'NUM_CLASSES'):
            value_set.add(cfg.val_dataset_class.NUM_CLASSES)

        if len(value_set) == 0:
            raise ValueError(
                '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
            )
        elif len(value_set) > 1:
            raise ValueError(
                '`num_classes` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(value_set))

        model_config[value_name] = value_set.pop()


class DefaultSyncImgChannelsRule(Rule):
    def check_and_correct(self, cfg):
        model_config = cfg.dic['model']
        train_dataset_config = cfg.dic['train_dataset']
        val_dataset_config = cfg.dic['val_dataset']
        value_set = set()

        # If the model has backbone, in_channels is the input params of backbone.
        # Otherwise, in_channels is the input params of the model.
        if 'backbone' in model_config:
            x = model_config['backbone'].get('in_channels', None)
            if x is not None:
                value_set.add(x)
        if 'in_channels' in model_config:
            value_set.add(model_config['in_channels'])
        if 'img_channels' in train_dataset_config:
            value_set.add(train_dataset_config['img_channels'])
        if 'img_channels' in val_dataset_config:
            value_set.add(val_dataset_config['img_channels'])
        if hasattr(cfg.train_dataset_class, 'IMG_CHANNELS'):
            value_set.add(cfg.train_dataset_class.IMG_CHANNELS)
        if hasattr(cfg.val_dataset_class, 'IMG_CHANNELS'):
            value_set.add(cfg.val_dataset_class.IMG_CHANNELS)

        if len(value_set) > 1:
            raise ValueError(
                '`in_channels` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(value_set))
        channels = 3 if len(value_set) == 0 else value_set.pop()

        if 'backbone' in model_config:
            model_config['backbone']['in_channels'] = channels
        else:
            model_config['in_channels'] = channels


class DefaultSyncIgnoreIndexRule(Rule):
    def __init__(self, loss_name):
        super().__init__()
        self.loss_name = loss_name

    def check_and_correct(self, cfg):
        def _check_ignore_index(loss_cfg, dataset_ignore_index):
            if 'ignore_index' in loss_cfg:
                assert loss_cfg['ignore_index'] == dataset_ignore_index, \
                    'the ignore_index in loss and train_dataset must be the same. Currently, loss ignore_index = {}, '\
                    'train_dataset ignore_index = {}'.format(loss_cfg['ignore_index'], dataset_ignore_index)
            else:
                loss_cfg['ignore_index'] = dataset_ignore_index

        loss_cfg = cfg.dic.get(self.loss_name, None)
        if loss_cfg is None:
            return

        train_dataset_config = cfg.dic['train_dataset']
        val_dataset_config = cfg.dic['val_dataset']
        value_set = set()
        value_name = 'ignore_index'

        if value_name in train_dataset_config:
            value_set.add(train_dataset_config[value_name])
        if value_name in val_dataset_config:
            value_set.add(val_dataset_config[value_name])
        if hasattr(cfg.train_dataset_class, 'IGNORE_INDEX'):
            value_set.add(cfg.train_dataset_class.IGNORE_INDEX)
        if hasattr(cfg.val_dataset_class, 'IGNORE_INDEX'):
            value_set.add(cfg.val_dataset_class.IGNORE_INDEX)

        if len(value_set) > 1:
            raise ValueError(
                '`ignore_index` is not consistent: {}. Please set it consistently in train_dataset and val_dataset'
                .format(value_set))
        ignore_index = 255 if len(value_set) == 0 else value_set.pop()

        for loss_cfg_i in loss_cfg['types']:
            if loss_cfg_i['type'] == 'MixedLoss':
                for loss_cfg_j in loss_cfg_i['losses']:
                    _check_ignore_index(loss_cfg_j, ignore_index)
            else:
                _check_ignore_index(loss_cfg_i, ignore_index)
