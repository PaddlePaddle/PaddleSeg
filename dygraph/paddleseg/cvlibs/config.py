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

import codecs
import os
from typing import Any, Callable

import yaml
import paddle

import paddleseg.cvlibs.manager as manager


class Config(object):
    '''
    Training config.

    Args:
        path(str) : the path of config file, supports yaml format only
    '''

    def __init__(self,
                 path: str,
                 learning_rate: float = None,
                 batch_size: int = None,
                 iters: int = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None
        self._losses = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate, batch_size=batch_size, iters=iters)

    def _update_dic(self, dic, base_dic):
        """
        update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: float = None,
               batch_size: int = None,
               iters: int = None):
        '''Update config'''
        if learning_rate:
            self.dic['learning_rate']['value'] = learning_rate

        if batch_size:
            self.dic['batch_size'] = batch_size

        if iters:
            self.dic['iters'] = iters

    @property
    def batch_size(self) -> int:
        return self.dic.get('batch_size', 1)

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        if not iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters

    @property
    def learning_rate(self) -> paddle.optimizer._LRScheduler:
        _learning_rate = self.dic.get('learning_rate', {}).get('value')
        if not _learning_rate:
            raise RuntimeError(
                'No learning rate specified in the configuration file.')

        args = self.decay_args
        decay_type = args.pop('type')

        if decay_type == 'poly':
            lr = _learning_rate
            return paddle.optimizer.PolynomialLR(lr, **args)
        else:
            raise RuntimeError('Only poly decay support.')

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        args = self.optimizer_args
        optimizer_type = args.pop('type')

        if optimizer_type == 'sgd':
            lr = self.learning_rate
            return paddle.optimizer.Momentum(
                lr, parameters=self.model.parameters(), **args)
        else:
            raise RuntimeError('Only sgd optimizer support.')

    @property
    def optimizer_args(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)

        return args

    @property
    def decay_args(self) -> dict:
        args = self.dic.get('learning_rate', {}).get('decay', {
            'type': 'poly',
            'power': 0.9
        }).copy()

        if args['type'] == 'poly':
            args.setdefault('decay_steps', self.iters)
            args.setdefault('end_lr', 0)

        return args

    @property
    def loss(self) -> dict:
        args = self.dic.get('loss', {}).copy()

        if not self._losses:
            self._losses = dict()
            for key, val in args.items():
                if key == 'types':
                    self._losses['types'] = []
                    for item in args['types']:
                        item['ignore_index'] = self.train_dataset.ignore_index
                        self._losses['types'].append(self._load_object(item))
                else:
                    self._losses[key] = val
            if len(self._losses['coef']) != len(self._losses['types']):
                raise RuntimeError(
                    'The length of coef should equal to types in loss config: {} != {}.'
                    .format(
                        len(self._losses['coef']), len(self._losses['types'])))
        return self._losses

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        model_cfg['num_classes'] = self.train_dataset.num_classes

        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')
        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.dic.get('train_dataset').copy()
        if not _train_dataset:
            return None
        return self._load_object(_train_dataset)

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.dic.get('val_dataset').copy()
        if not _val_dataset:
            return None
        return self._load_object(_val_dataset)

    def _load_component(self, com_name: str) -> Any:
        com_list = [
            manager.MODELS, manager.BACKBONES, manager.DATASETS,
            manager.TRANSFORMS, manager.LOSSES
        ]

        for com in com_list:
            if com_name in com.components_dict:
                return com[com_name]
        else:
            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, cfg: dict) -> Any:
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))

        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val

        return component(**params)

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self.dic)
