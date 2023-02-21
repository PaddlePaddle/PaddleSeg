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

from ..base import BaseModel
from ..base.utils.path import get_cache_dir
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath


class SegModel(BaseModel):
    _DUMMY_DATASET_DIR = os.path.join(get_cache_dir(), 'ppseg_dummy_dataset')

    def train(self,
              dataset=None,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              device='gpu',
              resume_path=None,
              dy2st=False,
              amp=None,
              use_vdl=True,
              ips=None,
              save_dir=None):
        if dataset is not None:
            # NOTE: We must use an absolute path here, 
            # so we can run the scripts either inside or outside the repo dir.
            dataset = abspath(dataset)
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'train'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(dataset)
        if dy2st:
            config.update({'to_static_training': True})
        config_path = self._config_path
        config.dump(config_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if resume_path is not None:
            model_dir = os.path.dirname(resume_path)
            cli_args.append(CLIArgument('--resume_path', model_dir))
        if amp is not None:
            if amp != 'OFF':
                cli_args.append(CLIArgument('--precision', 'fp16'))
                cli_args.append(CLIArgument('--amp_level', amp))
        if use_vdl:
            cli_args.append(CLIArgument('--use_vdl', '', sep=''))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        return self.runner.train(config_path, cli_args, device, ips)

    def evaluate(self,
                 weight_path,
                 dataset=None,
                 batch_size=None,
                 device='gpu',
                 amp=None,
                 ips=None):
        weight_path = abspath(weight_path)
        if dataset is not None:
            dataset = abspath(dataset)

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(dataset)
        config_path = self._config_path
        config.dump(config_path)

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if batch_size is not None:
            if batch_size != 1:
                raise ValueError("Batch size other than 1 is not supported.")
        if amp is not None:
            if amp != 'OFF':
                cli_args.append(CLIArgument('--precision', 'fp16'))
                cli_args.append(CLIArgument('--amp_level', amp))

        return self.runner.evaluate(config_path, cli_args, device, ips)

    def predict(self, weight_path, input_path, device='gpu', save_dir=None):
        weight_path = abspath(weight_path)
        input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'predict'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(self._create_dummy_dataset())
        config_path = self._config_path
        config.dump(config_path)

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if input_path is not None:
            cli_args.append(CLIArgument('--image_path', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        return self.runner.predict(config_path, cli_args, device)

    def export(self, weight_path, save_dir=None, input_shape=None):
        weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'export'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(self._create_dummy_dataset())
        config_path = self._config_path
        config.dump(config_path)

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

        return self.runner.export(config_path, cli_args, None)

    def infer(self, model_dir, input_path, device='gpu', save_dir=None):
        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'infer'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(self._create_dummy_dataset())
        config.dump(self._config_path)

        # Parse CLI arguments
        cli_args = []
        config_path = os.path.join(model_dir, 'deploy.yaml')
        if input_path is not None:
            cli_args.append(CLIArgument('--image_path', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        return self.runner.infer(config_path, cli_args, device)

    def compression(self,
                    weight_path,
                    dataset=None,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device='gpu',
                    use_vdl=True,
                    save_dir=None,
                    input_shape=None):
        weight_path = abspath(weight_path)
        if dataset is not None:
            dataset = abspath(dataset)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'compress'))

        # Update YAML config file
        # NOTE: In PaddleSeg, QAT does not use a different config file than regular training
        # Reusing `self.config` preserves the config items modified by the user when 
        # `SegModel` is initialized with a `SegConfig` object.
        config = self.config.copy()
        config.update_dataset(dataset)
        config_path = self._config_path
        config.dump(config_path)

        # Parse CLI arguments
        train_cli_args = []
        export_cli_args = []
        if batch_size is not None:
            train_cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            train_cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            train_cli_args.append(CLIArgument('--iters', epochs_iters))
        if weight_path is not None:
            train_cli_args.append(CLIArgument('--model_path', weight_path))
        if use_vdl:
            train_cli_args.append(CLIArgument('--use_vdl', '', sep=''))
        if save_dir is not None:
            train_cli_args.append(CLIArgument('--save_dir', save_dir))
            # The exported model saved in a subdirectory named `export`
            export_cli_args.append(
                CLIArgument('--save_dir', os.path.join(save_dir, 'export')))
        else:
            # According to
            # https://github.com/PaddlePaddle/PaddleSeg/blob/ba7b4d61e456fa8bfdfb7415c64cdce2945919d4/deploy/slim/quant/qat_train.py#L66
            save_dir = 'output'
        if input_shape is not None:
            if isinstance(input_shape, (list, tuple)):
                input_shape = ' '.join(map(str, input_shape))
            export_cli_args.append(
                CLIArgument(
                    '--input_shape', input_shape, sep=' '))

        return self.runner.compression(config_path, train_cli_args,
                                       export_cli_args, device, save_dir)

    def _create_dummy_dataset(self):
        # Create a PaddleSeg-style dataset
        # We will use this fake dataset to pass the config checks of PaddleSeg
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
