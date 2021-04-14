from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use gpu or cpu to test.")
    parser.add_argument(
        '--example', type=str, help='RoadLine, HumanSeg or ACE2P')

    return parser.parse_args()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


def merge_cfg_from_args(args, cfg):
    """Merge config keys, values in args into the global config."""
    for k, v in vars(args).items():
        d = cfg
        try:
            value = eval(v)
        except:
            value = v
        if value is not None:
            cfg[k] = value
