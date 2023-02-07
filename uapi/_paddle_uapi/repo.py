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

import sys
import abc
import contextlib

from . import primitives as P
from .utils import run_cmd as _run_cmd, CachedProperty as cached_property, abspath
from .path import create_yaml_config_file


class BaseRepo(metaclass=abc.ABCMeta):
    _ARGS_COMM_KEY = 'args'
    _CFG_FILE_COMM_KEY = 'cfg_file'
    _DEVICE_COMM_KEY = 'device'
    _MODEL_META_COMM_KEY = 'model_meta'
    _DATASET_META_COMM_KEY = 'dataset_meta'

    def __init__(self, root_path):
        self.root_path = abspath(root_path)

        self.python = sys.executable

        self.package_name_dict = {
            'shapely': 'Shapely',
            'cython': 'Cython',
            'pyyaml': 'PyYAML',
            'pillow': 'Pillow'
        }

        self.comm = None

    @abc.abstractmethod
    def check(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, comm):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, comm):
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, comm):
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, comm):
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, comm):
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_dataset(self, mode):
        raise NotImplementedError

    @abc.abstractmethod
    def _parse_config(self, cfg, mode):
        raise NotImplementedError

    def _check_environment(self, requirement_file_list=None,
                           requir_packages=[]):
        cmd = '{} -m pip freeze > installed_requirements.txt'.format(
            self.python)
        self.run_cmd(cmd)
        installed_packages = []
        for line in open('installed_requirements.txt').readlines():
            line = line.strip()
            if '==' in line:
                name, _ = line.split('==')
            elif ' @ ' in line:
                name, _ = line.split(' @ ')
            else:
                name = line
            installed_packages.append(name.strip())
        if requirement_file_list is not None:
            for file in requirement_file_list:
                requir_packages.extend(self._load_requirements(file))

        need_install = False
        for package in requir_packages:
            if package in self.package_name_dict:
                package = self.package_name_dict[package]
            if package not in installed_packages:
                need_install = True
        return need_install

    def _load_requirements(self, requirement_list):
        if isinstance(requirement_list, str):
            requirement_list = [requirement_list]
        requir_packages = []
        for file in requirement_list:
            for line in open(file).readlines():
                line = line.strip()
                if line.startswith('#'):
                    continue
                comparator_list = ['<=', '<', '==', '>=', '>']
                if True in [x in line for x in comparator_list]:
                    for comparator in comparator_list:
                        if comparator in line:
                            name, _ = line.split(comparator)
                            break
                else:
                    name = line
                name = name.strip()
                if len(name):
                    requir_packages.append(name)
        return requir_packages

    def distributed(self, comm):
        # TODO: docstring
        device = comm[self._DEVICE_COMM_KEY]
        python = self.python
        if device is None:
            return python
        dev_descs = device.split(',')
        num_devices = len(dev_descs)
        dev_ids = []
        for dev_desc in dev_descs:
            idx = dev_desc.find(':')
            if idx != -1:
                dev_ids.append(dev_desc[idx + 1:])
        dev_ids = ','.join(dev_ids)
        if num_devices > 1:
            python += " -m paddle.distributed.launch"
            if len(dev_ids) > 0:
                python += f" --gpus {dev_ids}"
        elif len(dev_ids) == 1:
            python = f"CUDA_VISIBLE_DEVICES={dev_ids} {python}"
        return python

    @contextlib.contextmanager
    def use_config(self, cfg, mode):
        self.set_config_file(cfg, mode)

        comm = cfg.comm

        # Copy config file to cache
        P.do_copy_file(cfg.config_file_path, self._CFG_FILE_PATH, comm=comm)

        # For efficiency, modify `cfg` in place
        comm[self._ARGS_COMM_KEY] = []
        comm[self._CFG_FILE_COMM_KEY] = self._CFG_FILE_PATH
        comm[self._MODEL_META_COMM_KEY] = cfg.model_info
        comm[self._DATASET_META_COMM_KEY] = getattr(cfg, 'dataset_info', None)

        _old_comm = self.comm
        self.comm = comm
        self._prepare_dataset(mode)
        self._parse_config(cfg, mode)

        try:
            yield comm
        finally:
            self.comm = _old_comm

    def add_cli_arg(self, arg_key, arg_val, sep=' ', comm=None):
        if comm is None:
            comm = self.comm
        return P.do_append_cli_arg(
            comm=comm,
            key=arg_key,
            val=arg_val,
            sep=sep,
            comm_key=self._ARGS_COMM_KEY)

    def modify_yaml_cfg(self,
                        cfg_desc,
                        val,
                        match_strategy='in_dict',
                        yaml_loader=None,
                        comm=None):
        if comm is None:
            comm = self.comm
        return P.do_modify_yaml_cfg_val(
            comm=comm,
            cfg_desc=cfg_desc,
            val=val,
            comm_key=self._CFG_FILE_COMM_KEY,
            match_strategy=match_strategy,
            yaml_loader=yaml_loader)

    def set_config_file(self, cfg, mode):
        if mode in ('train', 'predict', 'export', 'infer'):
            cfg.set_config_file(cfg.model_info['config_path'])
        elif mode == 'compress':
            cfg.set_config_file(cfg.model_info['auto_compression_config_path'])

    def run_cmd(self, cmd, switch_wdir=False, **kwargs):
        if switch_wdir:
            if 'wd' in kwargs:
                raise KeyError
            kwargs['wd'] = self.root_path
        return _run_cmd(cmd, **kwargs)

    @cached_property
    def _CFG_FILE_PATH(self):
        cls = self.__class__
        tag = cls.__name__.lower()
        # Allow overwriting
        return create_yaml_config_file(tag=tag, noclobber=False)
