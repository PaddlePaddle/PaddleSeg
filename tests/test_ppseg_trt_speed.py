import sys
import os
import copy
import unittest

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

from paddleseg.cvlibs import Config
from deploy.python.infer_onnx_trt import export_load_infer, parse_args
"""
Test the speed of different version PPSeg by ONNX and TRT.

For preparement and usage, please refer to `PaddleSeg/deploy/python/infer_onnx_trt.py`.
"""


def test_ppseg(args):

    test_model_cfg = [
        {},
        {
            'feat_nums': [2, 2, 2]
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_WeightedAdd_Add'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_ChAttenAdd0_Add'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_ChAttenAdd1_Add'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_ChAttenAdd2_Add'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_ChAttenAdd3_Add'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_ChAttenAdd0'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_ChAttenAdd1'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_ChAttenAdd2'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_ChAttenAdd3'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd0'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd1'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd2'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd3'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd4'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_Add_SpAttenAdd5'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_WeightedAdd0_SpAttenAdd1'
        },
        {
            'feat_nums': [2, 2, 2],
            'arm_type': 'ARM_WeightedAdd1_SpAttenAdd1'
        },
    ]

    base_cfg = Config(args.config)
    '''
    args.warmup = 10
    args.repeats = 20
    '''

    latency_list = []

    for item in test_model_cfg:
        test_cfg = copy.deepcopy(base_cfg)
        for key, val in item.items():
            assert key in test_cfg.dic['model'], \
                f"The model config does not have {key}."
            test_cfg.dic['model'][key] = val

        latency = export_load_infer(args, test_cfg.model)
        latency_list.append(latency)

        print("\n\n")

    for cfg, latency in zip(test_model_cfg, latency_list):
        cfg = 'base' if len(cfg) == 0 else cfg
        print("model config: {} , {:.3f} ms".format(str(cfg), latency))


if __name__ == '__main__':
    args = parse_args()
    test_ppseg(args)
