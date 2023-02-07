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

from .paddle_uapi.repo import BaseRepo
from .config import SegConfig


class PaddleSeg(BaseRepo):
    def check(self, model_name):
        # TODO:
        pass

    def build_repo_config(self, config_file_path=None):
        repo_config = SegConfig()
        if config_file_path is not None:
            repo_config.load(config_file_path)
        return repo_config

    def train(self, config_file_path, cli_args, device):
        python, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} tools/train.py --do_eval --config {config_file_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_file_path, cli_args, device):
        _, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/predict.py --config {config_file_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_file_path, cli_args, device):
        # `device` unused
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/export.py --config {config_file_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_file_path, cli_args, device):
        _, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} deploy/python/infer.py --config {config_file_path} --device {device_type} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def compression(self, config_file_path, cli_args, device):
        python, device_type = self.distributed(device)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} deploy/slim/quant/qat_train.py --do_eval --config {config_file_path} --device {device_type} {args_str}"
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
