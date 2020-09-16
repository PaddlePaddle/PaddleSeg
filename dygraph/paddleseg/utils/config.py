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
import pprint

import yaml
import paddle
import paddle.nn.functional as F

import paddleseg.cvlibs.manager as manager
from paddleseg.utils import logger


class Config(object):
    '''
    Training config.

    Args:
        path(str) : the path of config file, supports yaml format only
    '''

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        if path.endswith('yml') or path.endswith('yaml'):
            dic = self._parse_from_yaml(path)
            logger.info('\n' + pprint.pformat(dic))
            self._build(dic)
        else:
            raise RuntimeError('Config file should in yaml format!')

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

    def _build(self, dic: dict):
        '''Build config from dictionary'''
        dic = dic.copy()

        self._batch_size = dic.get('batch_size', 1)
        self._iters = dic.get('iters')

        if 'model' not in dic:
            raise RuntimeError()
        self._model_cfg = dic['model']
        self._model = None

        self._train_dataset = dic.get('train_dataset')
        self._val_dataset = dic.get('val_dataset')

        self._learning_rate_cfg = dic.get('learning_rate', {})
        self._learning_rate = self._learning_rate_cfg.get('value')
        self._decay = self._learning_rate_cfg.get('decay', {
            'type': 'poly',
            'power': 0.9
        })

        self._loss_cfg = dic.get('loss', {})
        self._losses = None

        self._optimizer_cfg = dic.get('optimizer', {})

    def update(self,
               learning_rate: float = None,
               batch_size: int = None,
               iters: int = None):
        '''Update config'''
        if learning_rate:
            self._learning_rate = learning_rate

        if batch_size:
            self._batch_size = batch_size

        if iters:
            self._iters = iters

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def iters(self) -> int:
        if not self._iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return self._iters

    @property
    def learning_rate(self) -> float:
        if not self._learning_rate:
            raise RuntimeError(
                'No learning rate specified in the configuration file.')

        if self.decay_type == 'poly':
            lr = self._learning_rate
            args = self.decay_args
            args.setdefault('decay_steps', self.iters)
            return paddle.optimizer.PolynomialLR(lr, **args)
        else:
            raise RuntimeError('Only poly decay support.')

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        if self.optimizer_type == 'sgd':
            lr = self.learning_rate
            args = self.optimizer_args
            args.setdefault('momentum', 0.9)
            return paddle.optimizer.Momentum(
                lr, parameters=self.model.parameters(), **args)
        else:
            raise RuntimeError('Only sgd optimizer support.')

    @property
    def optimizer_type(self) -> str:
        otype = self._optimizer_cfg.get('type')
        if not otype:
            raise RuntimeError(
                'No optimizer type specified in the configuration file.')
        return otype

    @property
    def optimizer_args(self) -> dict:
        args = self._optimizer_cfg.copy()
        args.pop('type')
        return args

    @property
    def decay_type(self) -> str:
        return self._decay['type']

    @property
    def decay_args(self) -> dict:
        args = self._decay.copy()
        args.pop('type')
        return args

    @property
    def loss(self) -> list:
        if not self._losses:
            args = self._loss_cfg.copy()
            self._losses = dict()
            for key, val in args.items():
                if key == 'types':
                    self._losses['types'] = []
                    for item in args['types']:
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
    def model(self) -> Callable:
        if not self._model:
            self._model = self._load_object(self._model_cfg)
        return self._model

    @property
    def train_dataset(self) -> Any:
        if not self._train_dataset:
            return None
        return self._load_object(self._train_dataset)

    @property
    def val_dataset(self) -> Any:
        if not self._val_dataset:
            return None
        return self._load_object(self._val_dataset)

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
