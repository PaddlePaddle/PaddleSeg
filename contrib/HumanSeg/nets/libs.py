# coding: utf8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
        for d in range(10):
            for t in [d, -d]:
                if G + t <= 0: continue
                if C % (G + t) == 0:
                    G = G + t
                    break
            if C % G == 0:
                break
    assert C % G == 0, "group can not divide channle"
    x = fluid.layers.group_norm(
        input,
        groups=G,
        param_attr=param_attr,
        bias_attr=bias_attr,
        name=name_scope + 'group_norm')
    return x


def bn(*args,
       norm_type='bn',
       eps=1e-5,
       bn_momentum=0.99,
       group_norm=32,
       **kargs):

    if norm_type == 'bn':
        with scope('BatchNorm'):
            return fluid.layers.batch_norm(
                *args,
                epsilon=eps,
                momentum=bn_momentum,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=bn_regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=bn_regularizer),
                moving_mean_name=name_scope + 'moving_mean',
                moving_variance_name=name_scope + 'moving_variance',
                **kargs)
    elif norm_type == 'gn':
        with scope('GroupNorm'):
            return group_norm(
                args[0],
                group_norm,
                eps=eps,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=bn_regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=bn_regularizer))
    else:
        raise Exception("Unsupport norm type:" + norm_type)


def bn_relu(data, norm_type='bn', eps=1e-5):
    return fluid.layers.relu(bn(data, norm_type=norm_type, eps=eps))


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


def separate_conv(input,
                  channel,
                  stride,
                  filter,
                  dilation=1,
                  act=None,
                  eps=1e-5):
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
        input = bn(input, eps=eps)
        if act: input = act(input)

    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('pointwise'):
        input = conv(
            input, channel, 1, 1, groups=1, padding=0, param_attr=param_attr)
        input = bn(input, eps=eps)
        if act: input = act(input)
    return input


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  if_act=True,
                  name=None,
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=fluid.ParamAttr(name=name + '_weights'),
        bias_attr=False)
    bn_name = name + '_bn'
    bn = fluid.layers.batch_norm(
        input=conv,
        param_attr=fluid.ParamAttr(name=bn_name + "_scale"),
        bias_attr=fluid.ParamAttr(name=bn_name + "_offset"),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')
    if if_act:
        return fluid.layers.relu6(bn)
    else:
        return bn


def sigmoid_to_softmax(input):
    """
    one channel to two channel
    """
    logit = fluid.layers.sigmoid(input)
    logit_back = 1 - logit
    logit = fluid.layers.concat([logit_back, logit], axis=1)
    return logit
