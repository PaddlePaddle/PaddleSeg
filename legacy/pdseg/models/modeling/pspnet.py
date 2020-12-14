# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import avg_pool, conv, bn
from models.backbone.resnet import ResNet as resnet_backbone
from utils.config import cfg


def get_logit_interp(input, num_classes, out_shape, name="logit"):
    # 根据类别数决定最后一层卷积输出, 并插值回原始尺寸
    param_attr = fluid.ParamAttr(
        name=name + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))

    with scope(name):
        logit = conv(
            input,
            num_classes,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=True,
            name=name + '_conv')
        logit_interp = fluid.layers.resize_bilinear(
            logit, out_shape=out_shape, name=name + '_interp')
    return logit_interp


def psp_module(input, out_features):
    # Pyramid Scene Parsing 金字塔池化模块
    # 输入：backbone输出的特征
    # 输出：对输入进行不同尺度pooling, 卷积操作后插值回原始尺寸，并concat
    #       最后进行一个卷积及BN操作

    cat_layers = []
    sizes = (1, 2, 3, 6)
    for size in sizes:
        psp_name = "psp" + str(size)
        with scope(psp_name):
            pool = fluid.layers.adaptive_pool2d(
                input,
                pool_size=[size, size],
                pool_type='avg',
                name=psp_name + '_adapool')
            data = conv(
                pool,
                out_features,
                filter_size=1,
                bias_attr=True,
                name=psp_name + '_conv')
            data_bn = bn(data, act='relu')
            interp = fluid.layers.resize_bilinear(
                data_bn, out_shape=input.shape[2:], name=psp_name + '_interp')
        cat_layers.append(interp)
    cat_layers = [input] + cat_layers[::-1]
    cat = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')

    psp_end_name = "psp_end"
    with scope(psp_end_name):
        data = conv(
            cat,
            out_features,
            filter_size=3,
            padding=1,
            bias_attr=True,
            name=psp_end_name)
        out = bn(data, act='relu')

    return out


def resnet(input):
    # PSPNET backbone: resnet, 默认resnet50
    # end_points: resnet终止层数
    # dilation_dict: resnet block数及对应的膨胀卷积尺度
    scale = cfg.MODEL.PSPNET.DEPTH_MULTIPLIER
    layers = cfg.MODEL.PSPNET.LAYERS
    end_points = layers - 1
    dilation_dict = {2: 2, 3: 4}
    model = resnet_backbone(layers, scale, stem='pspnet')
    data, _ = model.net(
        input, end_points=end_points, dilation_dict=dilation_dict)

    return data


def pspnet(input, num_classes):
    # Backbone: ResNet
    res = resnet(input)
    # PSP模块
    psp = psp_module(res, 512)
    dropout = fluid.layers.dropout(psp, dropout_prob=0.1, name="dropout")
    # 根据类别数决定最后一层卷积输出, 并插值回原始尺寸
    logit = get_logit_interp(dropout, num_classes, input.shape[2:])
    return logit
