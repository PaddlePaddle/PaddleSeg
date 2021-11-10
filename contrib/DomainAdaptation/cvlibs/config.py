# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import yaml
import paddle
import codecs
from typing import Any, Dict, Generic

from paddleseg.cvlibs import manager
from paddleseg.utils import logger
from models import EMA


class Config(object):
    '''
    Training configuration parsing. The only yaml/yml file is supported.

    The following hyper-parameters are available in the config file:
        batch_size: The number of samples per gpu.
        iters: The total training steps.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to paddleseg.datasets.
            For specific transforms, please refer to paddleseg.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently PaddleSeg only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        loss: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,
            where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of
            model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of
            loss type must be consistent with coef.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to paddleseg.models.
            For backbone, please refer to paddleseg.models.backbones.

    Args:
        path (str) : The path of config file, supports yaml format only.

    Examples:

        from paddleseg.cvlibs.config import Config

        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)

        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset

        # the argument of model should be parsed after dataset,
        # since the model builder uses some properties in dataset.
        model = cfg.model
        ...
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
        self.logger = logger
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate, batch_size=batch_size, iters=iters)

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

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
            if 'lr_scheduler' in self.dic:
                self.dic['lr_scheduler']['learning_rate'] = learning_rate
            else:
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
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            raise RuntimeError(
                'No `lr_scheduler` specified in the configuration file.')
        params = self.dic.get('lr_scheduler')

        lr_type = params.pop('type')
        if lr_type == 'PolynomialDecay':
            params.setdefault('decay_steps', self.iters)
            # params.setdefault('decay_steps', 400000)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)

        return getattr(paddle.optimizer.lr, lr_type)(**params)

    @property
    def learning_rate(self) -> paddle.optimizer.lr.LRScheduler:
        logger.warning(
            '''`learning_rate` in configuration file will be deprecated, please use `lr_scheduler` instead. E.g
            lr_scheduler:
                type: PolynomialDecay
                learning_rate: 0.01''')
        _learning_rate = self.dic.get('learning_rate', {}).get('value')
        if not _learning_rate:
            raise RuntimeError(
                'No learning rate specified in the configuration file.')

        args = self.decay_args
        decay_type = args.pop('type')

        if decay_type == 'poly':
            lr = _learning_rate
            return paddle.optimizer.lr.PolynomialDecay(lr, **args)
        elif decay_type == 'piecewise':
            values = _learning_rate
            return paddle.optimizer.lr.PiecewiseDecay(values=values, **args)
        elif decay_type == 'stepdecay':
            lr = _learning_rate
            return paddle.optimizer.lr.StepDecay(lr, **args)
        else:
            raise RuntimeError('Only poly and piecewise decay support.')

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        if 'lr_scheduler' in self.dic:
            lr = self.lr_scheduler
        else:
            lr = self.learning_rate
        args = self.optimizer_args
        optimizer_type = args.pop('type')

        if optimizer_type == 'sgd':
            return paddle.optimizer.Momentum(
                lr, parameters=self.model.parameters(), **args)
            # lr, parameters=self.model.backbone.optim_parameters(lr=lr.base_lr), **args)
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type in paddle.optimizer.__all__:
            return getattr(paddle.optimizer, optimizer_type)(
                lr, parameters=self.model.parameters(), **args)

        raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))

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
        if self._losses is None:
            self._losses = self._prepare_loss('loss')
        return self._losses

    @property
    def distill_loss(self) -> dict:
        if not hasattr(self, '_distill_losses'):
            self._distill_losses = self._prepare_loss('distill_loss')
        return self._distill_losses

    def _prepare_loss(self, loss_name):
        """
        Parse the loss parameters and load the loss layers.

        Args:
            loss_name (str): The root name of loss in the yaml file.
        Returns:
            dict: A dict including the loss parameters and layers.
        """
        args = self.dic.get(loss_name, {}).copy()
        if 'types' in args and 'coef' in args:
            len_types = len(args['types'])
            len_coef = len(args['coef'])
            if len_types != len_coef:
                if len_types == 1:
                    args['types'] = args['types'] * len_coef
                else:
                    raise ValueError(
                        'The length of types should equal to coef or equal to 1 in loss config, but they are {} and {}.'
                        .format(len_types, len_coef))
        else:
            raise ValueError(
                'Loss config should contain keys of "types" and "coef"')

        losses = dict()
        for key, val in args.items():
            if key == 'types':
                losses['types'] = []
                for item in args['types']:
                    if item['type'] != 'MixedLoss':
                        if 'ignore_index' in item:
                            assert item['ignore_index'] == self.train_dataset.ignore_index, 'If ignore_index of loss is set, '\
                            'the ignore_index of loss and train_dataset must be the same. \nCurrently, loss ignore_index = {}, '\
                            'train_dataset ignore_index = {}. \nIt is recommended not to set loss ignore_index, so it is consistent with '\
                            'train_dataset by default.'.format(item['ignore_index'], self.train_dataset.ignore_index)
                        item['ignore_index'] = \
                            self.train_dataset.ignore_index
                    losses['types'].append(self._load_object(item))
            else:
                losses[key] = val
        if len(losses['coef']) != len(losses['types']):
            raise RuntimeError(
                'The length of coef should equal to types in loss config: {} != {}.'
                .format(len(losses['coef']), len(losses['types'])))
        return losses

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

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

    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self.dic)
