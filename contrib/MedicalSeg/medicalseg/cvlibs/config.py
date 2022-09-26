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

import codecs
import os
import warnings
from typing import Any, Dict, Generic

import paddle
import yaml

from medicalseg.cvlibs import manager
from medicalseg.utils import logger

# todo: check and edit the unnecessary components


class Config(object):
    '''
    Parse training configuration. Only supports yaml/yml file.

    The following hyper-parameters are available in the config file:
        batch_size: The number of samples per gpu.
        iters: The total training steps.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to medseg.datasets.
            For specific transforms, please refer to medseg.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, currently medseg only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        loss: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,
            where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of
            model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of
            loss type must be consistent with coef.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to medseg.models.
            For backbone, please refer to medseg.models.backbones.

    Args:
        path (str) : The path of config file, only supports yaml format.

    Examples:

        from medseg.cvlibs.config import Config

        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)

        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset

        # model builder uses some properties in dataset
        # so model argument should be parsed after dataset.
        model = cfg.model
        ...
    '''

    def __init__(self,
                 path: str,
                 learning_rate: float=None,
                 batch_size: int=None,
                 iters: int=None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
            self.data_root_path_warning()
        else:
            raise RuntimeError('Config file should in yaml format!')

        self._model = None
        self._losses = None

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
               learning_rate: float=None,
               batch_size: int=None,
               iters: int=None):
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

        _learning_rate = self.dic.get('learning_rate', {})
        if isinstance(_learning_rate, float):
            return _learning_rate

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
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type in paddle.optimizer.__all__:
            return getattr(paddle.optimizer,
                           optimizer_type)(lr,
                                           parameters=self.model.parameters(),
                                           **args)

        raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))

    @property
    def optimizer_args(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)

        return args

    @property
    def decay_args(self) -> dict:
        args = self.dic.get('learning_rate', {}).get(
            'decay', {'type': 'poly',
                      'power': 0.9}).copy()

        if args['type'] == 'poly':
            args.setdefault('decay_steps', self.iters)
            args.setdefault('end_lr', 0)

        return args

    @property
    def loss(self) -> dict:
        if self._losses is None:
            self._losses = self._prepare_loss('loss')
        return self._losses

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
        if not 'num_classes' in model_cfg:
            num_classes = None
            if self.train_dataset_config:
                if hasattr(self.train_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.train_dataset_class.NUM_CLASSES
                elif hasattr(self.train_dataset, 'num_classes'):
                    num_classes = self.train_dataset.num_classes
            elif self.val_dataset_config:
                if hasattr(self.val_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.val_dataset_class.NUM_CLASSES
                elif hasattr(self.val_dataset, 'num_classes'):
                    num_classes = self.val_dataset.num_classes

            if num_classes is not None:
                model_cfg['num_classes'] = num_classes

        if not self._model:
            self._model = self._load_object(model_cfg)
        if paddle.get_device() != 'cpu':
            self._model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self._model)

        return self._model

    @property
    def train_dataset_config(self) -> Dict:
        trainset_config = self.dic.get('train_dataset', {}).copy()

        trainset_config['dataset_root'] = os.path.join(
            self.dic['data_root'], trainset_config.get('dataset_root'))
        trainset_config['result_dir'] = os.path.join(
            self.dic['data_root'], trainset_config.get('result_dir'))
        return trainset_config

    @property
    def val_dataset_config(self) -> Dict:
        valset_config = self.dic.get('val_dataset', {}).copy()

        valset_config['dataset_root'] = os.path.join(
            self.dic['data_root'], valset_config.get('dataset_root'))
        valset_config['result_dir'] = os.path.join(
            self.dic['data_root'], valset_config.get('result_dir'))
        return valset_config

    @property
    def train_dataset_class(self) -> Generic:
        dataset_type = self.train_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def val_dataset_class(self) -> Generic:
        dataset_type = self.val_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        return self._load_object(_train_dataset)

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        return self._load_object(_val_dataset)

    @property
    def test_dataset_config(self) -> Dict:
        testset_config = self.dic.get('test_dataset', {}).copy()

        testset_config['dataset_root'] = os.path.join(
            self.dic['data_root'], testset_config.get('dataset_root'))
        testset_config['result_dir'] = os.path.join(
            self.dic['data_root'], testset_config.get('result_dir'))
        return testset_config

    @property
    def test_dataset_class(self) -> Generic:
        dataset_type = self.test_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def test_dataset(self) -> paddle.io.Dataset:
        _test_dataset = self.test_dataset_config
        if not _test_dataset:
            return None
        return self._load_object(_test_dataset)

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
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    @property
    def to_static_training(self) -> bool:
        '''Whether to use @to_static for training'''
        return self.dic.get('to_static_training', False)

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self.dic)

    def data_root_path_warning(self):
        if "data_root" not in self.dic:
            raise RuntimeError('The dataroot need to be set in the config file')

        data_root = self.dic["data_root"]
        absolute_data_dir = os.path.join(os.getcwd(), data_root)
        if data_root == 'data/':
            warnings.warn(
                "Warning: The data dir now is {}, you should change the data_root in the global.yml if this directory didn\'t have enough space"
                .format(absolute_data_dir))
