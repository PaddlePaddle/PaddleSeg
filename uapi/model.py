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

import os

from .base_uapi import BaseModel
from .base_uapi.utils.path import get_cache_dir
from .base_uapi.utils.arg import CLIArgument
from .base_uapi.utils.misc import abspath
from .config import SegConfig


class SegModel(BaseModel):
    _DUMMY_DATASET_DIR = os.path.join(get_cache_dir(), 'ppseg_dummy_dataset')

    def build_repo_config(self, config_file_path=None):
        repo_config = SegConfig()
        if config_file_path is not None:
            repo_config.load(config_file_path)
        return repo_config

    def train(self,
              dataset,
              batch_size=None,
              epochs_iters=None,
              device=None,
              resume_path=None,
              dy2st=None,
              amp=None,
              save_dir=None):
        # We must use an absolute path here, 
        # so we can run the scripts either inside or outsize the repo dir.
        dataset = abspath(dataset)
        if dy2st is None:
            dy2st = False
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        repo_config = self.build_repo_config(config_file_path)
        repo_config.update_dataset_config(dataset)
        config_file_path = self.config_file_path
        repo_config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if epochs_iters is not None:
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if resume_path is not None:
            model_dir = os.path.dirname(resume_path)
            cli_args.append(CLIArgument('--resume_path', model_dir))
        # `dy2st` ignored
        if amp is not None:
            cli_args.append(CLIArgument('--precision', 'fp16'))
            cli_args.append(CLIArgument('--amp_level', amp))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.repo_instance.train(config_file_path, cli_args, device)

    def predict(self,
                weight_path=None,
                device=None,
                input_path=None,
                save_dir=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        repo_config = self.build_repo_config(config_file_path)
        repo_config.update_dataset_config(self._create_dummy_dataset())
        config_file_path = self.config_file_path
        repo_config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if input_path is not None:
            cli_args.append(CLIArgument('--image_path', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.repo_instance.predict(config_file_path, cli_args, device)

    def export(self, weight_path=None, save_dir=None, input_shape=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        repo_config = self.build_repo_config(config_file_path)
        repo_config.update_dataset_config(self._create_dummy_dataset())
        config_file_path = self.config_file_path
        repo_config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))
        if input_shape is not None:
            if isinstance(input_shape, (list, tuple)):
                input_shape = ' '.join(map(str, input_shape))
            cli_args.append(CLIArgument('--input_shape', input_shape, sep=' '))

        self.repo_instance.export(config_file_path, cli_args, None)

    def infer(self, model_dir, device=None, input_path=None, save_dir=None):
        model_dir = abspath(model_dir)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        repo_config = self.build_repo_config(config_file_path)
        repo_config.update_dataset_config(self._create_dummy_dataset())
        repo_config.dump(self.config_file_path)

        # Parse CLI arguments
        cli_args = []
        config_file_path = os.path.join(model_dir, 'deploy.yaml')
        if input_path is not None:
            cli_args.append(CLIArgument('--image_path', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.repo_instance.infer(config_file_path, cli_args, device)

    def compression(self,
                    dataset,
                    batch_size=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['auto_compression_config_path']
        repo_config = self.build_repo_config(config_file_path)
        repo_config.update_dataset_config(dataset)
        config_file_path = self.config_file_path
        repo_config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if epochs_iters is not None:
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.repo_instance.compression(config_file_path, cli_args, device)

    def _create_dummy_dataset(self):
        # Create a PaddleSeg-style dataset
        dir_ = os.path.abspath(self._DUMMY_DATASET_DIR)
        if os.path.exists(dir_):
            return dir_
        else:
            os.makedirs(dir_)
            with open(os.path.join(dir_, 'train.txt'), 'w') as f:
                f.write('fake_train_im_path fake_train_label_path')
            with open(os.path.join(dir_, 'val.txt'), 'w') as f:
                f.write('fake_val_im_path fake_val_label_path')
            return dir_
