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

from .path import create_yaml_config_file
from .register import get_registered_model_info, build_repo_from_model_info
from .utils import CachedProperty as cached_property


class PaddleModel(metaclass=abc.ABCMeta):
    def __init__(self, model_name):
        self.name = model_name
        self.model_info = get_registered_model_info(model_name)
        self.repo_instance = build_repo_from_model_info(self.model_info)

    @abc.abstractmethod
    def train(self, dataset, batch_size, epochs_iters, device, resume_path,
              dy2st, amp, save_dir):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, weight_path, device, input_path, save_dir):
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, weight_path, save_dir, input_shape):
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model_dir, device, input_path, save_dir):
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, dataset, batch_size, epochs_iters, device,
                    weight_path, save_dir):
        raise NotImplementedError

    @cached_property
    def config_file_path(self):
        cls = self.__class__
        tag = cls.__name__.lower()
        # Allow overwriting
        return create_yaml_config_file(tag=tag, noclobber=False)
