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

import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Mapping, Sequence

import yaml

from qinspector.utils.logger import setup_logger

logger = setup_logger('config')


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


class ConfigParser(object):
    def __init__(self, args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print('args: ', args)
        self.model_cfg, self.env_cfg = self.merge_cfg(args, cfg)

    def merge_cfg(self, args, cfg):
        env_cfg = cfg['ENV']
        model_cfg = cfg['PipeLine']

        def merge(cfg, arg):
            merge_cfg = copy.deepcopy(cfg)
            for k, v in cfg.items():
                if k in arg:
                    merge_cfg[k] = arg[k]
                else:
                    if isinstance(v, dict):
                        merge_cfg[k] = merge(v, arg)
            return merge_cfg

        def merge_opt(cfg, arg):
            for k, v in arg.items():
                if isinstance(cfg, Sequence):
                    k = eval(k)
                    cfg[k] = merge_opt(cfg[k], v)
                else:
                    if (k in cfg and (isinstance(cfg[k], Sequence) or
                                      isinstance(cfg[k], Mapping)) and
                            isinstance(arg[k], Mapping)):
                        merge_opt(cfg[k], arg[k])
                    else:
                        cfg[k] = arg[k]
            return cfg

        args_dict = vars(args)
        for k, v in args_dict.items():
            if k not in env_cfg:
                env_cfg[k] = v
        env_cfg = merge(env_cfg, args_dict)
        print('env_cfg: ', env_cfg)
        if 'opt' in args_dict.keys() and args_dict['opt']:
            opt_dict = args_dict['opt']
            if opt_dict.get('ENV', None):
                env_cfg = merge_opt(env_cfg, opt_dict['ENV'])
            if opt_dict.get('PipeLine', None):
                model_cfg = merge_opt(model_cfg, opt_dict['PipeLine'])

        return model_cfg, env_cfg

    def parse(self):
        return self.model_cfg, self.env_cfg

    def print_cfg(self):
        print('----------- Environment Arguments -----------')
        buffer = yaml.dump(self.env_cfg)
        print(buffer)
        print('------------- Model Arguments ---------------')
        buffer = yaml.dump(self.model_cfg)
        print(buffer)
        print('---------------------------------------------')
