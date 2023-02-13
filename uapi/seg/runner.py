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

from ..base import BaseRunner


class SegRunner(BaseRunner):
    def train(self, config_path, cli_args, device):
        python, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} tools/train.py --do_eval --config {config_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_path, cli_args, device):
        _, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/predict.py --config {config_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path, cli_args, device):
        # `device` unused
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/export.py --config {config_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device):
        _, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} deploy/python/infer.py --config {config_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        # Step 1: Train model
        python, device_type = self.distributed(device)
        train_args = self._gather_opts_args(train_cli_args)
        train_args_str = ' '.join(str(arg) for arg in train_args)
        cmd = f"{python} deploy/slim/quant/qat_train.py --do_eval --config {config_path} --device {device_type} {train_args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        # Step 2: Export model
        export_args = self._gather_opts_args(export_cli_args)
        export_args_str = ' '.join(str(arg) for arg in export_args)
        # We export the best model on the validation dataset
        weight_path = os.path.join(train_save_dir, 'best_model',
                                   'model.pdparams')
        cmd = f"{self.python} deploy/slim/quant/qat_export.py --config {config_path} --model_path {weight_path} {export_args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

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
