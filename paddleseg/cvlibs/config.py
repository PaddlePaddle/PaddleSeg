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

import six
import codecs
import os
from ast import literal_eval
from typing import Any, Dict, Optional

import yaml
import paddle

from . import _config_checkers as checker
from . import builder
from paddleseg.cvlibs import manager
from paddleseg.utils import logger, utils
from paddleseg.utils.utils import CachedProperty as cached_property

_INHERIT_KEY = '_inherited_'
_BASE_KEY = '_base_'


class Config(object):
    """
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
    """

    def __init__(self,
                 path: str,
                 learning_rate: Optional[float]=None,
                 batch_size: Optional[int]=None,
                 iters: Optional[int]=None,
                 opts: Optional[list]=None,
                 sanity_checker: Optional[checker.ConfigChecker]=None,
                 component_builder: Optional[builder.ComponentBuilder]=None):
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)

        self.dic = self.parse_from_yaml(path)
        self.dic = self.update_config_dict(
            self.dic,
            learning_rate=learning_rate,
            batch_size=batch_size,
            iters=iters,
            opts=opts)

        # We have to build the component builder before doing any sanity checks
        # This is because during a sanity check, some component objects are (possibly) 
        # required to be constructed.
        if component_builder is None:
            component_builder = self._build_default_component_builder()
        self.builder = component_builder

        if sanity_checker is None:
            sanity_checker = self._build_default_sanity_checker()
        sanity_checker.apply_all_rules(self)

        self._model = None
        self._losses = None

    def __str__(self) -> str:
        # Use NoAliasDumper to avoid yml anchor 
        return yaml.dump(self.dic, Dumper=utils.NoAliasDumper)

    #################### hyper parameters
    @cached_property
    def batch_size(self) -> int:
        return self.dic.get('batch_size', 1)

    @cached_property
    def iters(self) -> int:
        iters = self.dic.get('iters', None)
        if iters is None:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters

    @cached_property
    def to_static_training(self) -> bool:
        '''Whether to use @to_static for training'''
        return self.dic.get('to_static_training', False)

    #################### lr_scheduler and optimizer
    @cached_property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        assert 'lr_scheduler' in self.dic, 'No `lr_scheduler` specified in the configuration file.'
        params = self.dic.get('lr_scheduler')

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
            iters = self.iters - warmup_iters if use_warmup else self.iters
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

    @cached_property
    def optimizer_config(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        # TODO remove the default params
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)
        return args

    @cached_property
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
    @cached_property
    def loss(self) -> dict:
        if self._losses is None:
            self._losses = self._prepare_loss('loss')
        return self._losses

    @cached_property
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
            losses['types'].append(self.builder.create_object(loss_cfg))
        return losses

    #################### model
    @cached_property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not self._model:
            self._model = self.builder.create_object(model_cfg)
        return self._model

    #################### dataset
    @cached_property
    def train_dataset_config(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @cached_property
    def val_dataset_config(self) -> Dict:
        return self.dic.get('val_dataset', {}).copy()

    @cached_property
    def train_dataset_class(self) -> Any:
        dataset_type = self.train_dataset_config['type']
        return self.builder.load_component_class(dataset_type)

    @cached_property
    def val_dataset_class(self) -> Any:
        dataset_type = self.val_dataset_config['type']
        return self.builder.load_component_class(dataset_type)

    @cached_property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        return self.builder.create_object(_train_dataset)

    @cached_property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        return self.builder.create_object(_val_dataset)

    @cached_property
    def val_transforms(self) -> list:
        """Get val_transform from val_dataset"""
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return []
        _transforms = _val_dataset.get('transforms', [])
        transforms = []
        for tf in _transforms:
            transforms.append(self.builder.create_object(tf))
        return transforms

    #################### test and export
    @cached_property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    # TODO remove export_config
    @cached_property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    @classmethod
    def update_config_dict(cls, dic: dict, *args, **kwargs) -> dict:
        return _update_config_dict(dic, *args, **kwargs)

    @classmethod
    def parse_from_yaml(cls, path: str, *args, **kwargs) -> dict:
        return parse_from_yaml(path, *args, **kwargs)

    @classmethod
    def _build_default_sanity_checker(cls):
        rules = []
        rules.append(checker.DefaultPrimaryRule())
        rules.append(checker.DefaultSyncNumClassesRule())
        rules.append(checker.DefaultSyncImgChannelsRule())
        # Losses
        rules.append(checker.DefaultLossRule('loss'))
        rules.append(checker.DefaultSyncIgnoreIndexRule('loss'))
        # Distillation losses
        rules.append(checker.DefaultLossRule('distill_loss'))
        rules.append(checker.DefaultSyncIgnoreIndexRule('distill_loss'))

        return checker.ConfigChecker(rules, allow_update=True)

    @classmethod
    def _build_default_component_builder(cls):
        com_list = [
            manager.MODELS, manager.BACKBONES, manager.DATASETS,
            manager.TRANSFORMS, manager.LOSSES
        ]
        component_builder = builder.DefaultComponentBuilder(com_list=com_list)
        return component_builder


def merge_config_dicts(dic, base_dic):
    """Merge dic to base_dic and return base_dic."""
    base_dic = base_dic.copy()
    dic = dic.copy()

    if not dic.get(_INHERIT_KEY, True):
        dic.pop(_INHERIT_KEY)
        return dic

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = merge_config_dicts(val, base_dic[key])
        else:
            base_dic[key] = val

    return base_dic


def parse_from_yaml(path: str):
    """Parse a yaml file and build config"""
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if _BASE_KEY in dic:
        base_files = dic.pop(_BASE_KEY)
        if isinstance(base_files, str):
            base_files = [base_files]
        for bf in base_files:
            base_path = os.path.join(os.path.dirname(path), bf)
            base_dic = parse_from_yaml(base_path)
            dic = merge_config_dicts(dic, base_dic)

    return dic


def _update_config_dict(dic: dict,
                        learning_rate: Optional[float]=None,
                        batch_size: Optional[int]=None,
                        iters: Optional[int]=None,
                        opts: Optional[list]=None):
    """Update config"""
    # TODO: If the items to update are marked as anchors in the yaml file,
    # we should synchronize the references.
    dic = dic.copy()

    if learning_rate:
        dic['lr_scheduler']['learning_rate'] = learning_rate
    if batch_size:
        dic['batch_size'] = batch_size
    if iters:
        dic['iters'] = iters

    if opts is not None:
        for item in opts:
            assert ('=' in item) and (len(item.split('=')) == 2), "--opts params should be key=value," \
                " such as `--opts train.batch_size=1 test_config.scales=0.75,1.0,1.25`, " \
                "but got ({})".format(opts)

            key, value = item.split('=')
            if isinstance(value, six.string_types):
                try:
                    value = literal_eval(value)
                except:
                    # Ignore exceptions during literal evaluation
                    pass
            key_list = key.split('.')

            tmp_dic = dic
            for subkey in key_list[:-1]:
                assert subkey in tmp_dic, "Can not update {}, because it is not in config.".format(
                    key)
                tmp_dic = tmp_dic[subkey]
            tmp_dic[key_list[-1]] = value

    return dic
