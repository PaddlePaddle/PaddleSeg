# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import abc
import collections.abc
from collections import OrderedDict

__all__ = ['Config', 'BaseConfig']


class Config(object):
    _DICT_TYPE_ = OrderedDict

    def __init__(self, cfg=None):
        super().__init__()
        self._dict = self._DICT_TYPE_()
        if cfg is not None:
            # Manipulate the internal `_dict` such that we avoid an extra copy
            self.reset_from_dict(cfg._dict)

    @property
    def dict(self):
        return dict(self._dict)

    def __getattr__(self, key):
        try:
            val = self._dict[key]
            return val
        except KeyError:
            raise AttributeError

    def set_val(self, key, val):
        self._dict[key] = val

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val

    def __contains__(self, key):
        return key in self._dict

    def new_config(self, **kwargs):
        cfg = self.copy()
        cfg.update(kwargs)

    def copy(self):
        return type(self)(self)

    def pop(self, key):
        self._dict.pop(key)

    def __repr__(self):
        return format_cfg(self, indent=0)

    def reset_from_dict(self, dict_like_obj):
        self._dict.clear()
        self._dict.update(dict_like_obj)


class BaseConfig(Config, metaclass=abc.ABCMeta):
    """
    Abstract base class of Config.

    Config provides the funtionality to load, parse, or dump to a 
        configuration file with a specific format. Also, it provides 
        APIs to update configurations of several important 
        hyperparameters and model components.
    """

    @abc.abstractmethod
    def load(self, config_file_path):
        """Load configurations from a file."""
        raise NotImplementedError

    @abc.abstractmethod
    def dump(self, config_file_path):
        """Dump configurations to a file."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, dict_like_obj):
        """Update configurations from a dict-like object."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_dataset(self, dataset_path, dataset_type=None):
        """Update configurations of dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_optimizer(self, optimizer_type):
        """Update configurations of optimizer."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_backbone(self, backbone_type):
        """Update configurations of backbone."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_lr_scheduler(self, lr_scheduler_type):
        """Update configurations of lr scheduler."""
        raise NotImplementedError

    @abc.abstractmethod
    def update_batch_size(self, batch_size, mode='train'):
        """
        Update batch size. 
        
        By default this method modifies the training batch size.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_weight_decay(self, weight_decay):
        """Update configurations of weight decay."""
        raise NotImplementedError

    @classmethod
    def build_from_file(cls, config_file_path, *args, **kwargs):
        cfg = cls(*args, **kwargs)
        cfg.load(config_file_path)
        return cfg


def format_cfg(cfg, indent=0):
    MAP_TYPES = (collections.abc.Mapping, )
    SEQ_TYPES = (list, tuple)
    NESTED_TYPES = (*MAP_TYPES, *SEQ_TYPES)

    s = ' ' * indent
    if isinstance(cfg, Config):
        cfg = cfg.dict
    if isinstance(cfg, MAP_TYPES):
        for i, (k, v) in enumerate(sorted(cfg.items())):
            s += str(k) + ': '
            if isinstance(v, NESTED_TYPES):
                s += '\n' + format_cfg(v, indent=indent + 1)
            else:
                s += str(v)
            if i != len(cfg) - 1:
                s += '\n'
    elif isinstance(cfg, SEQ_TYPES):
        for i, v in enumerate(cfg):
            s += '- '
            if isinstance(v, NESTED_TYPES):
                s += '\n' + format_cfg(v, indent=indent + 1)
            else:
                s += str(v)
            if i != len(cfg) - 1:
                s += '\n'
    else:
        s += str(cfg)
    return s
