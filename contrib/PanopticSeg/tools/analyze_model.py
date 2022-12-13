#!/usr/bin/env python

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import sys
import contextlib

import paddle
from paddle.hapi.static_flops import Table
from paddle.hapi.dynamic_flops import (count_parameters, register_hooks,
                                       count_io_info)
from paddleseg.utils import get_sys_env, logger, op_flops_funs

from paddlepanseg.project_manager import work_on_project
from paddlepanseg.cvlibs import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Model analysis")
    parser.add_argument('--config', help="Config file.", type=str)
    parser.add_argument('--proj', help="Project name.", type=str)
    parser.add_argument(
        '--input_shape',
        nargs='+',
        type=int,
        help="Shape of the input shape, e.g. `--input_shape 1 3 1024 1024`",
        default=[1, 3, 1024, 1024])
    parser.add_argument(
        '--print_detail',
        help="To print detail information of each layer.",
        action='store_true')
    return parser.parse_args()


@contextlib.contextmanager
def _redirect_stdout_to_str(*args, **kwargs):
    with io.StringIO() as stdout:
        old_stdout = sys.stdout
        sys.stdout = stdout
        try:
            yield stdout
        finally:
            sys.stdout = old_stdout


def _dynamic_flops(model, inputs, custom_ops=None, print_detail=False):
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                print("Customize Function has been applied to {}.".format(
                    m_type))
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                print("{}'s FLOPs has been counted.".format(m_type))
        else:
            if m_type not in types_collection:
                print(
                    "Cannot find suitable count function for {}. Treat it as zero FLOPs."
                    .format(m_type))

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.framework.no_grad():
        model(inputs)

    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            total_ops += m.total_ops
            total_params += m.total_params

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    table = Table([
        "Layer Name", "Input Shape", "Output Shape", "Params (M)", "Flops (G)"
    ])

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            table.add_row([
                m.full_name(), list(m.input_shape.numpy()),
                list(m.output_shape.numpy()),
                round(float(m.total_params / 1e6), 3),
                round(float(m.total_ops / 1e9), 3)
            ])
            m._buffers.pop('total_ops')
            m._buffers.pop('total_params')
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if print_detail:
        with _redirect_stdout_to_str() as sio:
            table.print_table()
            tab_info = sio.getvalue()
        logger.info('\n' + tab_info)
    logger.info("Total FLOPs: {:.4f} G     Total Params: {:.4f} M".format(
        round(float(total_ops / 1e9), 3), round(float(total_params / 1e6), 3)))
    return int(total_ops)


def analyze(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    paddle.set_device('cpu')

    custom_ops = {paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn}
    inputs = paddle.randn(args.input_shape)

    cfg = Config(args.config)

    if args.proj is not None:
        with work_on_project(args.proj):
            _dynamic_flops(
                cfg.model,
                inputs,
                custom_ops=custom_ops,
                print_detail=args.print_detail)
    else:
        _dynamic_flops(
            cfg.model,
            inputs,
            custom_ops=custom_ops,
            print_detail=args.print_detail)


if __name__ == '__main__':
    args = parse_args()
    if not args.config:
        raise RuntimeError("No configuration file has been specified.")

    logger.info("config:" + args.config)
    logger.info("input_shape:")
    logger.info(args.input_shape)

    analyze(args)
