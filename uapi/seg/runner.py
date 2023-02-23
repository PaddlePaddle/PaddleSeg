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
import sys

from ..base import BaseRunner


class SegRunner(BaseRunner):
    def train(self, config_path, cli_args, device, ips):
        python = self.distributed(device, ips)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} tools/train.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def evaluate(self, config_path, cli_args, device, ips):
        python = self.distributed(device, ips)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{python} tools/val.py --config {config_path} {args_str}"
        cp = self.run_cmd(
            cmd,
            switch_wdir=True,
            echo=True,
            silent=False,
            pipe_stdout=True,
            pipe_stderr=True)

        if cp.returncode == 0:
            sys.stdout.write(cp.stdout)
            metric_dict = _extract_eval_metrics(cp.stdout)
            for k, v in metric_dict.items():
                setattr(cp, k, v)
        else:
            sys.stderr.write(cp.stderr)
            # XXX: This can get stuck in some cases?
            cp.err_info = cp.stderr
        return cp

    def predict(self, config_path, cli_args, device):
        # `device` unused
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/predict.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path, cli_args, device):
        # `device` unused
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} tools/export.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device):
        # `device` unused
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        cmd = f"{self.python} deploy/python/infer.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        # Step 1: Train model
        python = self.distributed(device)
        train_args = self._gather_opts_args(train_cli_args)
        train_args_str = ' '.join(str(arg) for arg in train_args)
        # Note that we add `--do_eval` here so we can have `train_save_dir/best_model/model.pdparams` saved
        cmd = f"{python} deploy/slim/quant/qat_train.py --do_eval --config {config_path} {train_args_str}"
        cp_train = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        # Step 2: Export model
        export_args = self._gather_opts_args(export_cli_args)
        export_args_str = ' '.join(str(arg) for arg in export_args)
        # We export the best model on the validation dataset
        weight_path = os.path.join(train_save_dir, 'best_model',
                                   'model.pdparams')
        cmd = f"{self.python} deploy/slim/quant/qat_export.py --config {config_path} --model_path {weight_path} {export_args_str}"
        cp_export = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        return cp_train, cp_export

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


def _extract_eval_metrics(stdout):
    import re

    _DP = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    pattern = r'Images: \d+ mIoU: (_dp) Acc: (_dp) Kappa: (_dp) Dice: (_dp)'.replace(
        '_dp', _DP)
    keys = ['miou', 'acc', 'kappa', 'dice']

    metric_dict = dict()
    pattern = re.compile(pattern)
    # TODO: Use lazy operation to reduce cost for long outputs
    lines = stdout.splitlines()
    for line in lines:
        match = pattern.search(line)
        if match:
            for k, v in zip(keys, map(float, match.groups())):
                metric_dict[k] = v
    return metric_dict
