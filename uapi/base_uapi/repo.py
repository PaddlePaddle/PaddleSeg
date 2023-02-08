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

from .utils.misc import run_cmd as _run_cmd, abspath


class BaseRepo(metaclass=abc.ABCMeta):
    def __init__(self, root_path):
        self.root_path = abspath(root_path)

        self.python = sys.executable

        self.package_name_dict = {
            'shapely': 'Shapely',
            'cython': 'Cython',
            'pyyaml': 'PyYAML',
            'pillow': 'Pillow'
        }

    @abc.abstractmethod
    def check(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, config_file_path, cli_args, device):
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

    def distributed(self, device):
        # TODO: docstring
        python = self.python
        if device is None:
            # By default use a GPU device
            return python, 'gpu'
        # According to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html
        if ':' not in device:
            return python, device
        else:
            device, dev_ids = device.split(':')
            num_devices = len(dev_ids.split(','))
        if num_devices > 1:
            python += " -m paddle.distributed.launch"
            python += f" --gpus {dev_ids}"
        elif num_devices == 1:
            # TODO: Accommodate Windows system
            python = f"CUDA_VISIBLE_DEVICES={dev_ids} {python}"
        return python, device

    def run_cmd(self, cmd, switch_wdir=False, **kwargs):
        if switch_wdir:
            if 'wd' in kwargs:
                raise KeyError
            kwargs['wd'] = self.root_path
        return _run_cmd(cmd, **kwargs)
