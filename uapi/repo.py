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

from .paddle_uapi.path import get_cache_dir
from .paddle_uapi.repo import BaseRepo


class PaddleSeg(BaseRepo):
    _INFER_CFG_FILE_COMM_KEY = 'infer_cfg_file'
    _DUMMY_DATASET_DIR = os.path.join(get_cache_dir(), 'ppseg_dummy_dataset')

    def check(self, model_name):
        # TODO:
        pass

    def train(self, comm):
        python = self.distributed(comm)
        args = self._gather_opts_args(comm[self._ARGS_COMM_KEY])
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} tools/train.py --do_eval --config={comm[self._CFG_FILE_COMM_KEY]} {args_str}"
        self.run_cmd(cmd, silent=False)

    def predict(self, comm):
        args = self._gather_opts_args(comm[self._ARGS_COMM_KEY])
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/predict.py --config={comm[self._CFG_FILE_COMM_KEY]} {args_str}"
        self.run_cmd(cmd, silent=False)

    def export(self, comm):
        args = self._gather_opts_args(comm[self._ARGS_COMM_KEY])
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/export.py --config={comm[self._CFG_FILE_COMM_KEY]} {args_str}"
        self.run_cmd(cmd, silent=False)

    def infer(self, comm):
        args = self._gather_opts_args(comm[self._ARGS_COMM_KEY])
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} deploy/python/infer.py --config={comm[self._INFER_CFG_FILE_COMM_KEY]} {args_str}"
        self.run_cmd(cmd, silent=False)

    def compression(self, comm):
        python = self.distributed(comm)
        args = self._gather_opts_args(comm[self._ARGS_COMM_KEY])
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} deploy/slim/quant/qat_train.py --do_eval --config={comm[self._CFG_FILE_COMM_KEY]} {args_str}"
        self.run_cmd(cmd, silent=False)

    def _gather_opts_args(self, args):
        # Since `--opts` in PaddleSeg does not use `action='append'`
        # We collect and arrange all opts args here
        # e.g.: python tools/train.py --config xxx --opts a=1 c=3 --opts b=2
        # => python tools/train.py --config xxx c=3 --opts a=1 b=2
        # NOTE: This is an inplace operation
        def _is_opts_arg(arg):
            return arg.key.lstrip().startswith('--opts')

        # We note that Python built-in `sorted()` preserves the order (stable)
        args = sorted(args, key=_is_opts_arg)
        found = False
        for arg in args:
            if _is_opts_arg(arg):
                if found:
                    arg.key = arg.key.replace('--opts', '')
                else:
                    # Found first
                    found = True

        return args

    def _parse_config(self, cfg, mode):
        for key, val in cfg.dict.items():
            # Common configs
            if key == 'batch_size':
                self.add_cli_arg('--batch_size', val)
            elif key == 'epochs_iters':
                self.add_cli_arg('--iters', val)
            elif key == 'device':
                device = val
                self.add_cli_arg('--device', device)
                cfg.comm[self._DEVICE_COMM_KEY] = device
            elif key == 'save_dir':
                self.add_cli_arg('--save_dir', val)
            elif key == 'weight_path':
                self.add_cli_arg('--model_path', val)
            if mode == 'train':
                # `dy2st` ignored
                if key.startswith('train_dataset.'):
                    self.add_cli_arg(f'--opts {key}', val, sep='=')
                elif key.startswith('val_dataset.'):
                    self.add_cli_arg(f'--opts {key}', val, sep='=')
                elif key == 'resume_path':
                    if val is not None:
                        weight_path = val
                        model_dir = os.path.dirname(weight_path)
                        self.add_cli_arg('--resume_path', model_dir)
                elif key == 'amp':
                    if val is not None:
                        self.add_cli_arg('--precision', 'fp16')
                        self.add_cli_arg('--amp_level', val)
            elif mode == 'predict':
                if key == 'input_path':
                    self.add_cli_arg('--image_path', val)
            elif mode == 'export':
                if key == 'input_shape':
                    if val is not None:
                        input_shape = val
                        if isinstance(input_shape, (list, tuple)):
                            input_shape = ' '.join(map(str, input_shape))
                        self.add_cli_arg('--input_shape', input_shape, sep=' ')
            elif mode == 'infer':
                if key == 'model_dir':
                    model_dir = val
                    cfg.comm[self._INFER_CFG_FILE_COMM_KEY] = os.path.join(
                        model_dir, 'deploy.yaml')
                elif key == 'input_path':
                    self.add_cli_arg('--image_path', val)
            elif mode == 'compress':
                if key.startswith('train_dataset.'):
                    self.add_cli_arg(f'--opts {key}', val, sep='=')
                elif key.startswith('val_dataset.'):
                    self.add_cli_arg(f'--opts {key}', val, sep='=')

    def _prepare_dataset(self, mode):
        dataset_meta = self.comm[self._DATASET_META_COMM_KEY]
        if dataset_meta is None:
            dataset_dir = self._create_dummy_dataset()
            self.modify_yaml_cfg('train_dataset.type', 'Dataset')
            self.modify_yaml_cfg('train_dataset.dataset_root', dataset_dir)
            self.modify_yaml_cfg('train_dataset.train_path',
                                 os.path.join(dataset_dir, 'train.txt'))
            self.modify_yaml_cfg('val_dataset.type', 'Dataset')
            self.modify_yaml_cfg('val_dataset.dataset_root', dataset_dir)
            self.modify_yaml_cfg('val_dataset.val_path',
                                 os.path.join(dataset_dir, 'val.txt'))
        else:
            dataset_dir = dataset_meta['dataset_root_dir']
            self.add_cli_arg('--opts train_dataset.type', 'Dataset', sep='=')
            self.add_cli_arg(
                '--opts train_dataset.dataset_root', dataset_dir, sep='=')
            self.add_cli_arg(
                '--opts train_dataset.train_path',
                os.path.join(dataset_dir, 'train.txt'),
                sep='=')
            self.add_cli_arg('--opts val_dataset.type', 'Dataset', sep='=')
            self.add_cli_arg(
                '--opts val_dataset.dataset_root', dataset_dir, sep='=')
            self.add_cli_arg(
                '--opts val_dataset.val_path',
                os.path.join(dataset_dir, 'val.txt'),
                sep='=')

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
