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
from models.libs.model_libs import scope
from models.libs.model_libs import bn, bn_relu, relu, conv_bn_layer
from models.libs.model_libs import conv, avg_pool
from models.libs.model_libs import separate_conv
from utils.config import cfg


def learning_to_downsample(x, dw_channels1=32, dw_channels2=48,
                           out_channels=64):
    x = relu(bn(conv(x, dw_channels1, 3, 2)))
    with scope('dsconv1'):
        x = separate_conv(
            x, dw_channels2, stride=2, filter=3, act=fluid.layers.relu)
    with scope('dsconv2'):
        x = separate_conv(
            x, out_channels, stride=2, filter=3, act=fluid.layers.relu)
    return x


def shortcut(input, data_residual):
    return fluid.layers.elementwise_add(input, data_residual)


def dropout2d(input, prob, is_train=False):
    if not is_train:
        return input
    channels = input.shape[1]
    keep_prob = 1.0 - prob
    shape = fluid.layers.shape(input)
    random_tensor = keep_prob + fluid.layers.uniform_random(
        [shape[0], channels, 1, 1], min=0., max=1.)
    binary_tensor = fluid.layers.floor(random_tensor)
    output = input / keep_prob * binary_tensor
    return output


def inverted_residual_unit(input,
                           num_in_filter,
                           num_filters,
                           ifshortcut,
                           stride,
                           filter_size,
                           padding,
                           expansion_factor,
                           name=None):
    num_expfilter = int(round(num_in_filter * expansion_factor))

    channel_expand = conv_bn_layer(
        input=input,
        num_filters=num_expfilter,
        filter_size=1,
        stride=1,
        padding=0,
        num_groups=1,
        if_act=True,
        name=name + '_expand')

    bottleneck_conv = conv_bn_layer(
        input=channel_expand,
        num_filters=num_expfilter,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        num_groups=num_expfilter,
        if_act=True,
        name=name + '_dwise',
        use_cudnn=False)

    depthwise_output = bottleneck_conv

    linear_out = conv_bn_layer(
        input=bottleneck_conv,
        num_filters=num_filters,
        filter_size=1,
        stride=1,
        padding=0,
        num_groups=1,
        if_act=False,
        name=name + '_linear')

    if ifshortcut:
        out = shortcut(input=input, data_residual=linear_out)
        return out, depthwise_output
    else:
        return linear_out, depthwise_output


def inverted_blocks(input, in_c, t, c, n, s, name=None):
    first_block, depthwise_output = inverted_residual_unit(
        input=input,
        num_in_filter=in_c,
        num_filters=c,
        ifshortcut=False,
        stride=s,
        filter_size=3,
        padding=1,
        expansion_factor=t,
        name=name + '_1')

    last_residual_block = first_block
    last_c = c

    for i in range(1, n):
        last_residual_block, depthwise_output = inverted_residual_unit(
            input=last_residual_block,
            num_in_filter=last_c,
            num_filters=c,
            ifshortcut=True,
            stride=1,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + '_' + str(i + 1))
    return last_residual_block, depthwise_output


def psp_module(input, out_features):

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
                bias_attr=False,
                name=psp_name + '_conv')
            data_bn = bn(data, act='relu')
            interp = fluid.layers.resize_bilinear(
                data_bn,
                out_shape=input.shape[2:],
                name=psp_name + '_interp',
                align_mode=0)
        cat_layers.append(interp)
    cat_layers = [input] + cat_layers
    out = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')

    return out


class FeatureFusionModule:
    """Feature fusion module"""

    def __init__(self,
                 higher_in_channels,
                 lower_in_channels,
                 out_channels,
                 scale_factor=4):
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

    def net(self, higher_res_feature, lower_res_feature):
        h, w = higher_res_feature.shape[2:]
        lower_res_feature = fluid.layers.resize_bilinear(
            lower_res_feature, [h, w], align_mode=0)

        with scope('dwconv'):
            lower_res_feature = relu(
                bn(conv(lower_res_feature, self.out_channels,
                        1)))  #(lower_res_feature)
        with scope('conv_lower_res'):
            lower_res_feature = bn(
                conv(lower_res_feature, self.out_channels, 1, bias_attr=True))
        with scope('conv_higher_res'):
            higher_res_feature = bn(
                conv(higher_res_feature, self.out_channels, 1, bias_attr=True))
        out = higher_res_feature + lower_res_feature

        return relu(out)


class GlobalFeatureExtractor():
    """Global feature extractor module"""

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 t=6,
                 num_blocks=(3, 3, 3)):
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.out_channels = out_channels
        self.t = t
        self.num_blocks = num_blocks

    def net(self, x):
        x, _ = inverted_blocks(x, self.in_channels, self.t,
                               self.block_channels[0], self.num_blocks[0], 2,
                               'inverted_block_1')
        x, _ = inverted_blocks(x, self.block_channels[0], self.t,
                               self.block_channels[1], self.num_blocks[1], 2,
                               'inverted_block_2')
        x, _ = inverted_blocks(x, self.block_channels[1], self.t,
                               self.block_channels[2], self.num_blocks[2], 1,
                               'inverted_block_3')
        x = psp_module(x, self.block_channels[2] // 4)
        with scope('out'):
            x = relu(bn(conv(x, self.out_channels, 1)))
        return x


class Classifier:
    """Classifier"""

    def __init__(self, dw_channels, num_classes, stride=1):
        self.dw_channels = dw_channels
        self.num_classes = num_classes
        self.stride = stride

    def net(self, x):
        with scope('dsconv1'):
            x = separate_conv(
                x,
                self.dw_channels,
                stride=self.stride,
                filter=3,
                act=fluid.layers.relu)
        with scope('dsconv2'):
            x = separate_conv(
                x,
                self.dw_channels,
                stride=self.stride,
                filter=3,
                act=fluid.layers.relu)

        x = dropout2d(x, 0.1, is_train=cfg.PHASE == 'train')
        x = conv(x, self.num_classes, 1, bias_attr=True)
        return x


def aux_layer(x, num_classes):
    x = relu(bn(conv(x, 32, 3, padding=1)))
    x = dropout2d(x, 0.1, is_train=(cfg.PHASE == 'train'))
    with scope('logit'):
        x = conv(x, num_classes, 1, bias_attr=True)
    return x


def fast_scnn(img, num_classes):
    size = img.shape[2:]
    classifier = Classifier(128, num_classes)

    global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6,
                                                      [3, 3, 3])
    feature_fusion = FeatureFusionModule(64, 128, 128)

    with scope('learning_to_downsample'):
        higher_res_features = learning_to_downsample(img, 32, 48, 64)
    with scope('global_feature_extractor'):
        lower_res_feature = global_feature_extractor.net(higher_res_features)
    with scope('feature_fusion'):
        x = feature_fusion.net(higher_res_features, lower_res_feature)
    with scope('classifier'):
        logit = classifier.net(x)
        logit = fluid.layers.resize_bilinear(logit, size, align_mode=0)

    if len(cfg.MODEL.MULTI_LOSS_WEIGHT) == 3:
        with scope('aux_layer_higher'):
            higher_logit = aux_layer(higher_res_features, num_classes)
            higher_logit = fluid.layers.resize_bilinear(
                higher_logit, size, align_mode=0)
        with scope('aux_layer_lower'):
            lower_logit = aux_layer(lower_res_feature, num_classes)
            lower_logit = fluid.layers.resize_bilinear(
                lower_logit, size, align_mode=0)
        return logit, higher_logit, lower_logit
    elif len(cfg.MODEL.MULTI_LOSS_WEIGHT) == 2:
        with scope('aux_layer_higher'):
            higher_logit = aux_layer(higher_res_features, num_classes)
            higher_logit = fluid.layers.resize_bilinear(
                higher_logit, size, align_mode=0)
        return logit, higher_logit

    return logit
