#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

import paddle
from paddle.hapi.static_flops import Table
from paddle.hapi.dynamic_flops import (count_parameters, register_hooks,
                                       count_io_info)
from paddleseg.utils import get_sys_env, logger, op_flops_funs, utils

from paddlepanseg.cvlibs import Config, make_default_builder


def parse_args():
    parser = argparse.ArgumentParser(description="Model analysis")
    parser.add_argument('--config', dest='cfg', help="Config file.", type=str)
    parser.add_argument(
        '--input_shape',
        nargs='+',
        type=int,
        help="Shape of the input shape, e.g. `--input_shape 1 3 1024 1024`",
        default=[1, 3, 1024, 1024])
    parser.add_argument(
        '--num_levels',
        type=int,
        help="Maximum levels of layers to show.",
        default=None)
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


def _count_layer_stats(layer, counters, level, res):
    info = OrderedDict()
    info['Layer Name'] = layer.full_name()
    info['Level'] = level
    children = list(layer.children())
    if len(children) > 0:
        children_names = set(m.full_name() for m in children)
        res_of_layer = []
        for child in children:
            res_of_layer = _count_layer_stats(child, counters, level + 1,
                                              res_of_layer)
        for name in counters.keys():
            info[name] = sum(item[name] for item in res_of_layer
                             if item['Layer Name'] in children_names)
        res.append(info)
        res.extend(res_of_layer)
    else:
        # XXX: Hard-code default items
        if hasattr(layer, 'input_shape'):
            info['Input Shape'] = layer.input_shape.numpy().tolist()
        if hasattr(layer, 'output_shape'):
            info['Output Shape'] = layer.output_shape.numpy().tolist()
        for name, cnter in counters.items():
            info[name] = cnter(layer)
        res.append(info)
    return res


def _stats_to_table(stats, cols):
    levels = set(info['Level'] for info in stats)
    min_level = min(levels)
    num_pad_cols = max(levels) - min_level
    # Assume that the first column is Layer Name
    cols = cols[:1] + [''] * num_pad_cols + cols[1:]
    table = Table(cols)
    for info in stats:
        level = info['Level']
        row = [info.get(key, '') for key in cols if key != '']
        # Round float numbers
        for i, ele in enumerate(row):
            if isinstance(ele, float):
                row[i] = _round(ele)
        rel_level = (level - min_level)
        row = [''] * rel_level + [row[0]] + [''] * (num_pad_cols - rel_level
                                                    ) + row[1:]
        table.add_row(row)
    return table


def _round(x, digits=3):
    return round(x, digits)


def _to_mega(x):
    return float(x / 1e6)


def _to_giga(x):
    return float(x / 1e9)


def dynamic_flops(model, inputs, custom_ops=None, num_levels=None):
    def _add_hooks(m):
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

    if num_levels is not None and num_levels < 1:
        raise ValueError("`num_levels` must be a positive integer.")

    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    training = model.training

    model.eval()
    model.apply(_add_hooks)

    with paddle.no_grad():
        model(inputs)

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    counters = {
        'Params (M)': lambda m: _to_mega(m.total_params),
        'FLOPs (G)': lambda m: _to_giga(m.total_ops)
    }
    stats = _count_layer_stats(model, counters, 1, [])
    if num_levels is not None:
        stats = list(filter(lambda info: info['Level'] <= num_levels, stats))
    table = _stats_to_table(
        stats, ['Layer Name', 'Input Shape', 'Output Shape', *counters.keys()])

    with _redirect_stdout_to_str() as sio:
        table.print_table()
        tab_info = sio.getvalue()
    logger.info('\n' + tab_info)


def analyze(args, cfg):
    custom_ops = {paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn}
    inputs = paddle.randn(args.input_shape)

    builder = make_default_builder(cfg)
    dynamic_flops(
        builder.model,
        inputs,
        custom_ops=custom_ops,
        num_levels=args.num_levels)


if __name__ == '__main__':
    args = parse_args()

    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")
    cfg = Config(args.cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    logger.info("input_shape:")
    logger.info(args.input_shape)

    paddle.set_device('cpu')

    analyze(args, cfg)
