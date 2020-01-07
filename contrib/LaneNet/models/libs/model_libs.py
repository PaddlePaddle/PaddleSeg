# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from utils.config import cfg
import contextlib

bn_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)
name_scope = ""


@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '/'
    yield
    name_scope = bk


def max_pool(input, kernel, stride, padding):
    data = fluid.layers.pool2d(
        input,
        pool_size=kernel,
        pool_type='max',
        pool_stride=stride,
        pool_padding=padding)
    return data


def avg_pool(input, kernel, stride, padding=0):
    data = fluid.layers.pool2d(
        input,
        pool_size=kernel,
        pool_type='avg',
        pool_stride=stride,
        pool_padding=padding)
    return data


def group_norm(input, G, eps=1e-5, param_attr=None, bias_attr=None):
    N, C, H, W = input.shape
    if C % G != 0:
        # print "group can not divide channle:", C, G
        for d in range(10):
            for t in [d, -d]:
                if G + t <= 0: continue
                if C % (G + t) == 0:
                    G = G + t
                    break
            if C % G == 0:
                # print "use group size:", G
                break
    assert C % G == 0
    x = fluid.layers.group_norm(
        input,
        groups=G,
        param_attr=param_attr,
        bias_attr=bias_attr,
        name=name_scope + 'group_norm')
    return x


def bn(*args, **kargs):
    if cfg.MODEL.DEFAULT_NORM_TYPE == 'bn':
        with scope('BatchNorm'):
            return fluid.layers.batch_norm(
                *args,
                epsilon=cfg.MODEL.DEFAULT_EPSILON,
                momentum=cfg.MODEL.BN_MOMENTUM,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=bn_regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=bn_regularizer),
                moving_mean_name=name_scope + 'moving_mean',
                moving_variance_name=name_scope + 'moving_variance',
                **kargs)
    elif cfg.MODEL.DEFAULT_NORM_TYPE == 'gn':
        with scope('GroupNorm'):
            return group_norm(
                args[0],
                cfg.MODEL.DEFAULT_GROUP_NUMBER,
                eps=cfg.MODEL.DEFAULT_EPSILON,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=bn_regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=bn_regularizer))
    else:
        raise Exception("Unsupport norm type:" + cfg.MODEL.DEFAULT_NORM_TYPE)


def bn_relu(data):
    return fluid.layers.relu(bn(data))


def relu(data):
    return fluid.layers.relu(data)


def conv(*args, **kargs):
    kargs['param_attr'] = name_scope + 'weights'
    if 'bias_attr' in kargs and kargs['bias_attr']:
        kargs['bias_attr'] = fluid.ParamAttr(
            name=name_scope + 'biases',
            regularizer=None,
            initializer=fluid.initializer.ConstantInitializer(value=0.0))
    else:
        kargs['bias_attr'] = False
    return fluid.layers.conv2d(*args, **kargs)


def deconv(*args, **kargs):
    kargs['param_attr'] = name_scope + 'weights'
    if 'bias_attr' in kargs and kargs['bias_attr']:
        kargs['bias_attr'] = name_scope + 'biases'
    else:
        kargs['bias_attr'] = False
    return fluid.layers.conv2d_transpose(*args, **kargs)


def separate_conv(input, channel, stride, filter, dilation=1, act=None):
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.33))
    with scope('depthwise'):
        input = conv(
            input,
            input.shape[1],
            filter,
            stride,
            groups=input.shape[1],
            padding=(filter // 2) * dilation,
            dilation=dilation,
            use_cudnn=False,
            param_attr=param_attr)
        input = bn(input)
        if act: input = act(input)

    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('pointwise'):
        input = conv(
            input, channel, 1, 1, groups=1, padding=0, param_attr=param_attr)
        input = bn(input)
        if act: input = act(input)
    return input
