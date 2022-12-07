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
import warnings
import six
from ast import literal_eval
from typing import Any, Dict, Generic

import paddle
import yaml

from paddleseg.cvlibs import manager
from paddleseg.utils import logger, utils


class Config(object):
    '''
    Training configuration parsing. The only yaml/yml file is supported.

    The following hyper-parameters are available in the config file:
        global: The global params in yml file, such as device, seed, etc.
        train: The params for training models in yml file, such as batch_size, iters, etc.
        test: The params for test models in yml file, such as scales, flip_horizontal, etc.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to paddleseg.datasets.
            For specific transforms, please refer to paddleseg.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently PaddleSeg only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        lr_scheduler: A learning rate scheduler. 
        loss: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,
            where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of
            model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of
            loss type must be consistent with coef.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to paddleseg.models.
            For backbone, please refer to paddleseg.models.backbones.

    Args:
        path (str) : The path of config file, supports yaml format only.
        opts (list, optional) : Use -o or --opts to update the key-value pairs in config file. Default: None 

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

    def __init__(self, path: str, opts: list=None):
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)

        self.dic = parse_from_yaml(path)
        self.dic = update_dic(self.dic, opts=opts)
        # TODO if it is old config, remind users to update config file

        self._check_config()
        self._check_sync_num_classes()
        self._check_sync_img_channels()
        self._check_sync_ignore_index('loss')
        self._check_sync_ignore_index('distill_loss')

        self._model = None
        self._losses = None

    def __str__(self) -> str:
        # Use NoAliasDumper to avoid yml anchor 
        return yaml.dump(self.dic, Dumper=utils.NoAliasDumper)

    #################### parameters
    def global_params(self, key):
        return self._get_params('global', key)

    def train_params(self, key):
        return self._get_params('train', key)

    def test_params(self, key):
        return self._get_params('test', key)

    def _get_params(self, params_name, key):
        assert params_name in self.dic, "{} is not in config file".format(
            params_name)
        assert key in self.dic[params_name], "{} is not in {}".format(
            key, params_name)
        return self.dic[params_name].get(key)

    #################### lr_scheduler and optimizer
    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        assert 'lr_scheduler' in self.dic, 'No `lr_scheduler` specified in the configuration file.'
        params = self.dic.get('lr_scheduler')

        # TODO refactor
        use_warmup = False
        if 'warmup_iters' in params:
            use_warmup = True
            warmup_iters = params.pop('warmup_iters')
            assert 'warmup_start_lr' in params, \
                "When use warmup, please set warmup_start_lr and warmup_iters in lr_scheduler"
            warmup_start_lr = params.pop('warmup_start_lr')
            end_lr = params['learning_rate']

        lr_type = params.pop('type')
        if lr_type == 'PolynomialDecay':
            iters = self.train_params('iters')
            iters = iters - warmup_iters if use_warmup else iters
            iters = max(iters, 1)
            params.setdefault('decay_steps', iters)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)
        lr_sche = getattr(paddle.optimizer.lr, lr_type)(**params)

        if use_warmup:
            lr_sche = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_iters,
                start_lr=warmup_start_lr,
                end_lr=end_lr)

        return lr_sche

    @property
    def optimizer_config(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        # TODO remove the default params
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)
        return args

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        lr = self.lr_scheduler
        args = self.optimizer_config
        optimizer_type = args.pop('type')

        # TODO refactor optimizer to support customized setting 
        params = self.model.parameters()
        if 'backbone_lr_mult' in args:
            if not hasattr(self.model, 'backbone'):
                logger.warning('The backbone_lr_mult is not effective because'
                               ' the model does not have backbone')
            else:
                backbone_lr_mult = args.pop('backbone_lr_mult')
                backbone_params = self.model.backbone.parameters()
                backbone_params_id = [id(x) for x in backbone_params]
                other_params = [
                    x for x in params if id(x) not in backbone_params_id
                ]
                params = [{
                    'params': backbone_params,
                    'learning_rate': backbone_lr_mult
                }, {
                    'params': other_params
                }]

        if optimizer_type == 'sgd':
            return paddle.optimizer.Momentum(lr, parameters=params, **args)
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(lr, parameters=params, **args)
        elif optimizer_type in paddle.optimizer.__all__:
            return getattr(paddle.optimizer, optimizer_type)(lr,
                                                             parameters=params,
                                                             **args)

        raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))

    #################### loss
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
        losses = {'coef': args['coef'], "types": []}
        for loss_cfg in args['types']:
            losses['types'].append(create_object(loss_cfg))
        return losses

    #################### model
    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not self._model:
            self._model = create_object(model_cfg)
        return self._model

    #################### dataset
    @property
    def train_dataset_config(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @property
    def val_dataset_config(self) -> Dict:
        return self.dic.get('val_dataset', {}).copy()

    @property
    def train_dataset_class(self) -> Generic:
        dataset_type = self.train_dataset_config['type']
        return load_component_class(dataset_type)

    @property
    def val_dataset_class(self) -> Generic:
        dataset_type = self.val_dataset_config['type']
        return load_component_class(dataset_type)

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        if not self.train_dataset_config:
            return None
        _train_dataset = create_object(self.train_dataset_config)
        _repeats = self.train_params("repeats")
        if _repeats > 1:
            _train_dataset.file_list *= _repeats
            logger.info("The images of train dataset is repeated by {} times".
                        format(_repeats))
        return _train_dataset

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        return create_object(_val_dataset)

    @property
    def val_transforms(self) -> list:
        """Get val_transform from val_dataset"""
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return []
        _transforms = _val_dataset.get('transforms', [])
        transforms = []
        for i in _transforms:
            transforms.append(create_object(i))
        return transforms

    #################### test and export
    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    #################### check and synchronize
    def _check_config(self):
        assert self.dic.get('model', None) is not None, \
            'No model specified in the configuration file.'
        assert self.train_dataset_config or self.val_dataset_config, \
            'One of `train_dataset` or `val_dataset should be given, but there are none.'

        def _check_loss_config(dic, loss_name):
            loss_cfg = dic.get(loss_name, None)
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
                        "The length of types should equal to coef or equal "
                        "to 1 in loss config, but they are {} and {}.".format(
                            len_types, len_coef))

        _check_loss_config(self.dic, 'loss')
        _check_loss_config(self.dic, 'distill_loss')

    def _check_sync_num_classes(self):
        """
        Check and sync num_classes between the model, train_dataset and val_dataset.
        """
        num_classes_set = set()

        if self.dic['model'].get('num_classes', None) is not None:
            num_classes_set.add(self.dic['model'].get('num_classes'))
        if self.train_dataset_config:
            if hasattr(self.train_dataset_class, 'NUM_CLASSES'):
                num_classes_set.add(self.train_dataset_class.NUM_CLASSES)
            if 'num_classes' in self.train_dataset_config:
                num_classes_set.add(self.train_dataset_config['num_classes'])
        if self.val_dataset_config:
            if hasattr(self.val_dataset_class, 'NUM_CLASSES'):
                num_classes_set.add(self.val_dataset_class.NUM_CLASSES)
            if 'num_classes' in self.val_dataset_config:
                num_classes_set.add(self.val_dataset_config['num_classes'])

        if len(num_classes_set) == 0:
            raise ValueError(
                '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
            )
        elif len(num_classes_set) > 1:
            raise ValueError(
                '`num_classes` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(num_classes_set))

        num_classes = num_classes_set.pop()
        self.dic['model']['num_classes'] = num_classes
        if self.train_dataset_config and \
            (not hasattr(self.train_dataset_class, 'NUM_CLASSES')):
            self.dic['train_dataset']['num_classes'] = num_classes
        if self.val_dataset_config and \
            (not hasattr(self.val_dataset_class, 'NUM_CLASSES')):
            self.dic['val_dataset']['num_classes'] = num_classes

    def _check_sync_img_channels(self):
        """
        Check and sync img_channels between the model, train_dataset and val_dataset.
        """
        img_channels_set = set()
        model_cfg = self.dic['model']

        # If the model has backbone, in_channels is the input params of backbone.
        # Otherwise, in_channels is the input params of the model.
        if 'backbone' in model_cfg:
            x = model_cfg['backbone'].get('in_channels', None)
            if x is not None:
                img_channels_set.add(x)
        elif model_cfg.get('in_channels', None) is not None:
            img_channels_set.add(model_cfg.get('in_channels'))
        if self.train_dataset_config and \
            ('img_channels' in self.train_dataset_config):
            img_channels_set.add(self.train_dataset_config['img_channels'])
        if self.val_dataset_config and \
            ('img_channels' in self.val_dataset_config):
            img_channels_set.add(self.val_dataset_config['img_channels'])

        if len(img_channels_set) > 1:
            raise ValueError(
                '`img_channels` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
                .format(img_channels_set))

        img_channels = 3 if len(img_channels_set) == 0 \
            else img_channels_set.pop()
        if 'backbone' in model_cfg:
            self.dic['model']['backbone']['in_channels'] = img_channels
        else:
            self.dic['model']['in_channels'] = img_channels
        if self.train_dataset_config and \
            self.train_dataset_config['type'] == "Dataset":
            self.dic['train_dataset']['img_channels'] = img_channels
        if self.val_dataset_config and \
            self.val_dataset_config['type'] == "Dataset":
            self.dic['val_dataset']['img_channels'] = img_channels

    def _check_sync_ignore_index(self, loss_name):
        """
        Check and sync ignore_index between the model, train_dataset and val_dataset.
        """
        loss_cfg = self.dic.get(loss_name, None)
        if loss_cfg is None:
            return

        def _check_ignore_index(loss_cfg, dataset_ignore_index):
            if 'ignore_index' in loss_cfg:
                assert loss_cfg['ignore_index'] == dataset_ignore_index, \
                    'the ignore_index in loss and train_dataset must be the same. Currently, loss ignore_index = {}, '\
                    'train_dataset ignore_index = {}'.format(loss_cfg['ignore_index'], dataset_ignore_index)

        dataset_ignore_index = self.train_dataset.ignore_index
        for loss_cfg_i in loss_cfg['types']:
            if loss_cfg_i['type'] == 'MixedLoss':
                for loss_cfg_j in loss_cfg_i['losses']:
                    _check_ignore_index(loss_cfg_j, dataset_ignore_index)
                    loss_cfg_j['ignore_index'] = dataset_ignore_index
            else:
                _check_ignore_index(loss_cfg_i, dataset_ignore_index)
                loss_cfg_i['ignore_index'] = dataset_ignore_index


def merge_config_dict(dic, base_dic):
    '''Merge dic to base_dic and return base_dic.'''
    base_dic = base_dic.copy()
    dic = dic.copy()

    if not dic.get('_inherited_', True):
        dic.pop('_inherited_')
        return dic

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = merge_config_dict(val, base_dic[key])
        else:
            base_dic[key] = val

    return base_dic


def parse_from_yaml(path: str):
    '''Parse a yaml file and build config'''
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if '_base_' in dic:
        base_files = dic.pop('_base_')
        if isinstance(base_files, str):
            base_files = [base_files]
        for bf in base_files:
            base_path = os.path.join(os.path.dirname(path), bf)
            base_dic = parse_from_yaml(base_path)
            dic = merge_config_dict(dic, base_dic)

    return dic


def update_dic(dic: dict, opts: list=None):
    '''Update config'''
    if opts is None:
        return dic

    dic = dic.copy()
    for item in opts:
        assert ('=' in item) and (len(item.split('=')) == 2), "--opts params should be key=value," \
            " such as `--opts train.batch_size=1 val.scales=0.75,1.0,1.25`, " \
            "but got ({})".format(item)

        key, value = item.split('=')
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except:
                pass
        key_list = key.split('.')

        tmp_dic = dic
        for subkey in key_list[:-1]:
            tmp_dic.setdefault(subkey, dict())
            tmp_dic = tmp_dic[subkey]
        tmp_dic[key_list[-1]] = value

    return dic


def load_component_class(com_name: str) -> Any:
    '''Load component class, such as model, loss, dataset, etc.'''
    com_list = [
        manager.MODELS, manager.BACKBONES, manager.DATASETS, manager.TRANSFORMS,
        manager.LOSSES
    ]

    for com in com_list:
        if com_name in com.components_dict:
            return com[com_name]

    raise RuntimeError('The specified component ({}) was not found.'.format(
        com_name))


def create_object(cfg: dict) -> Any:
    '''Create Python object, such as model, loss, dataset, etc.'''
    cfg = cfg.copy()
    if 'type' not in cfg:
        raise RuntimeError('No object information in {}.'.format(cfg))

    is_meta_type = lambda item: isinstance(item, dict) and 'type' in item
    component = load_component_class(cfg.pop('type'))

    params = {}
    for key, val in cfg.items():
        if is_meta_type(val):
            params[key] = create_object(val)
        elif isinstance(val, list):
            params[key] = [
                create_object(item) if is_meta_type(item) else item
                for item in val
            ]
        else:
            params[key] = val

    return component(**params)
