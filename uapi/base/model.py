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

from .register import (get_registered_model_info, build_runner_from_model_info,
                       build_model_from_model_info,
                       build_config_from_model_info)
from .utils.misc import CachedProperty as cached_property
from .utils.path import create_yaml_config_file


class PaddleModel(object):
    # We constrain function params here
    def __new__(cls, model_name=None, config=None):
        if model_name is None and config is None:
            raise ValueError(
                "At least one of `model_name` and `config` must be not None.")
        elif model_name is not None and config is not None:
            if model_name != config.model_name:
                raise ValueError(
                    "If both `model_name` and `config` are not None, `model_name` should be the same as `config.model_name`."
                )
        elif model_name is None and config is not None:
            model_name = config.model_name
        model_info = get_registered_model_info(model_name)
        return build_model_from_model_info(model_info, config=config)


class BaseModel(metaclass=abc.ABCMeta):
    """
    Abstract base class of Model.
    
    Model defines how Config and Runner interact with each other. In addition, Model 
        provides users with multiple APIs to perform model training, prediction, etc.

    Args:
        model_name (str): A registered model name.
        config (config.BaseConfig): Config.
    """

    def __init__(self, model_name, config):
        self.name = model_name
        self.model_info = get_registered_model_info(model_name)
        # NOTE: We build runner instance here by extracting runner info from model info
        # so that we don't have to overwrite the `__init__()` method of each child class.
        self.runner = build_runner_from_model_info(self.model_info)
        if config is None:
            config = build_config_from_model_info(self.model_info)
        self.config = config

    @abc.abstractmethod
    def train(self, dataset, batch_size, epochs_iters, device, resume_path,
              dy2st, amp, save_dir):
        """
        Train a model.

        Args:
            dataset (str|None): Root path of the dataset. If None, use a pre-defined default dataset.
            batch_size (int|None): Number of samples in each mini-batch. If None, use a pre-defined default 
                batch size.
            epochs_iters (int|None): Total iterations or epochs of model training. If None, use a 
                pre-defined default value of epochs/iterations.
            device (str|None): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'. 
            resume_path (str|None): If not None, resume training from the model snapshot stored in `resume_path`.
            dy2st (bool|None): Whether or not to enable dynamic-to-static training. If None, use a default 
                setting.
            amp (str|None): Optimization level to use in AMP training. Choices are ['O1', 'O2', None]. 
                If None, do not enable AMP training.
            save_dir (str|None): Directory to store model snapshots. If None, use a default setting. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, weight_path, input_path, device, save_dir):
        """
        Make prediction with a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            input_path (str): Path of the input file, e.g. an image.
            device (str|None): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'. 
            save_dir (str|None): Directory to store prediction results. If None, use a default setting. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, weight_path, save_dir):
        """
        Export a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            save_dir (str|None): Directory to store the exported model. If None, use a default setting. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model_dir, input_path, device, save_dir):
        """
        Make inference with an exported model.

        Args:
            model_dir (str): Path of the model snapshot to load.
            input_path (str): Path of the input file, e.g. an image.
            device (str|None): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'. 
            save_dir (str|None): Directory to store inference results. If None, use a default setting. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, weight_path, dataset, batch_size, epochs_iters,
                    device, save_dir):
        """
        Perform quantization aware training (QAT) and export the quantized model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            dataset (str|None): Root path of the dataset. If None, use a pre-defined default dataset.
            batch_size (int|None): Number of samples in each mini-batch. If None, use a pre-defined default 
                batch size.
            epochs_iters (int|None): Total iterations or epochs of model training. If None, use a 
                pre-defined default value of epochs/iterations.
            device (str|None): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'. 
            save_dir (str|None): Directory to store model snapshots. The exported model will be saved in the 
                `infer` subdirectory of `save_dir`. If None, use a default setting. 
        """
        raise NotImplementedError

    @cached_property
    def config_file_path(self):
        cls = self.__class__
        model_name = self.model_info['model_name']
        tag = '_'.join([cls.__name__.lower(), model_name])
        # Allow overwriting
        return create_yaml_config_file(tag=tag, noclobber=False)

    @cached_property
    def supported_apis(self):
        return tuple(self.model_info['supported_apis'])
