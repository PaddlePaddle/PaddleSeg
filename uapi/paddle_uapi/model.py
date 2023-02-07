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

import inspect

from .register import get_registered_model_info, build_repo_from_model_info
from .config import RepoBasicConfig
from .utils import switch_working_dir, abspath


class PaddleModel(object):
    def __init__(self, model_name):
        self.name = model_name
        # TODO: Extension of default config
        self.config = RepoBasicConfig()

        self.model_info = get_registered_model_info(model_name)
        self.repo_instance = build_repo_from_model_info(self.model_info)
        self.prepare_config_files()

    def prepare_config_files(self):
        # TODO:
        pass

    def train(self,
              dataset=None,
              batch_size=None,
              epochs_iters=None,
              device=None,
              resume_path=None,
              dy2st=None,
              amp=None,
              save_dir=None):
        if device is None:
            device = 'gpu'
        if dy2st is None:
            dy2st = False
        if dataset is not None:
            # We must use an absolute path here, 
            # so we can run the scripts either inside or outsize the repo dir.
            dataset = abspath(dataset)
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        config = self._extract_configs(self.train)
        config = self.config.new_config(**config.dict)
        config.model_info = self.model_info
        config.dataset_info = self.get_dataset_info(dataset)
        with switch_working_dir(self.repo_instance.root_path):
            with self.repo_instance.use_config(config, 'train') as comm:
                # Run train cmd
                self.repo_instance.train(comm)

    def predict(self,
                weight_path=None,
                device=None,
                input_path=None,
                save_dir=None):
        if device is None:
            device = 'gpu'
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        config = self._extract_configs(self.predict)
        config = self.config.new_config(**config.dict)
        config.model_info = self.model_info
        with switch_working_dir(self.repo_instance.root_path):
            with self.repo_instance.use_config(config, 'predict') as comm:
                # Run predict cmd
                self.repo_instance.predict(comm)

    def export(self, weight_path=None, save_dir=None, input_shape=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        config = self._extract_configs(self.export)
        config = self.config.new_config(**config.dict)
        config.model_info = self.model_info
        with switch_working_dir(self.repo_instance.root_path):
            with self.repo_instance.use_config(config, 'export') as comm:
                # Run export cmd
                self.repo_instance.export(comm)

    def infer(self, model_dir=None, device=None, input_path=None,
              save_dir=None):
        if device is None:
            device = 'gpu'
        if model_dir is not None:
            model_dir = abspath(model_dir)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        config = self._extract_configs(self.infer)
        config = self.config.new_config(**config.dict)
        config.model_info = self.model_info
        with switch_working_dir(self.repo_instance.root_path):
            with self.repo_instance.use_config(config, 'infer') as comm:
                # Run infer cmd
                self.repo_instance.infer(comm)

    def compression(self,
                    dataset=None,
                    batch_size=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        if device is None:
            device = 'gpu'
        if dataset is not None:
            dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        config = self._extract_configs(self.compression)
        config = self.config.new_config(**config.dict)
        config.model_info = self.model_info
        config.dataset_info = self.get_dataset_info(dataset)
        with switch_working_dir(self.repo_instance.root_path):
            with self.repo_instance.use_config(config, 'compress') as comm:
                # Run auto compression cmd
                self.repo_instance.compression(comm)

    def _extract_configs(self, bnd_method):
        sig = inspect.signature(bnd_method)
        config = RepoBasicConfig()
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                # Ignore params without a default value
                continue
            config[name] = param.default
        # Update config with local variables in the caller
        curr_frame = inspect.currentframe()
        prev_frame = curr_frame.f_back
        for var_name, var_val in prev_frame.f_locals.items():
            if var_name in config:
                config[var_name] = var_val
        return config

    def get_dataset_info(self, dataset):
        return {'dataset_root_dir': dataset}
