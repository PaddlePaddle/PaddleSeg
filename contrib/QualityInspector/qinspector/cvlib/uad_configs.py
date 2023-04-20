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

import yaml

from collections import namedtuple
from qinspector.utils.logger import setup_logger

logger = setup_logger('config')


class ConfigParser(object):
    '''
        ConfigParser is an object that can parse a yml config file.

    Args:
        args: An object of type argparse.
    '''

    def __init__(self, args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        self.cfg = self.merge_cfg(args, cfg)
        self.check_cfg()
        self.print_cfg()

    def merge_cfg(self, args, cfg):
        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                cfg[k] = v
        return cfg

    def check_cfg(self):
        device = self.cfg.get("device", "GPU")
        assert device.upper() in ['CPU', 'GPU', 'XPU'
                                  ], "device should be CPU, GPU or XPU"

    def parser(self):
        return Dic2Obj(self.cfg)

    def print_cfg(self):
        print('------------- Config Arguments ---------------')
        buffer = yaml.dump(self.cfg)
        print(buffer)
        print('---------------------------------------------')


class Dic2Obj(object):
    def __new__(cls, data):
        if isinstance(data, dict):
            return namedtuple('Dic2Obj',
                              data.keys())(*(Dic2Obj(val)
                                             for val in data.values()))
        elif isinstance(data, (tuple, list, set, frozenset)):
            return type(data)(Dic2Obj(_) for _ in data)
        else:
            return data
