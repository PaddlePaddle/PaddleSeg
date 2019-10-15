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
import contextlib
import paddle
import paddle.fluid as fluid
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, bn_relu, relu
from models.libs.model_libs import conv
from models.libs.model_libs import separate_conv
from models.backbone.mobilenet_v2 import MobileNetV2 as mobilenet_backbone
from models.backbone.xception import Xception as xception_backbone

def encoder(input):
    # 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
    # ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
    # OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
    # aspp_ratios：ASPP模块空洞卷积的采样率

    if cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 16:
        aspp_ratios = [6, 12, 18]
    elif cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 8:
        aspp_ratios = [12, 24, 36]
    else:
        raise Exception("deeplab only support stride 8 or 16")

    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('encoder'):
        channel = 256
        with scope("image_pool"):
            image_avg = fluid.layers.reduce_mean(
                input, [2, 3], keep_dim=True)
            image_avg = bn_relu(
                conv(
                    image_avg,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
            image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])

        with scope("aspp0"):
            aspp0 = bn_relu(
                conv(
                    input,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
        with scope("aspp1"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp1 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[0], act=relu)
            else:
                aspp1 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[0],
                        padding=aspp_ratios[0],
                        param_attr=param_attr))
        with scope("aspp2"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp2 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[1], act=relu)
            else:
                aspp2 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[1],
                        padding=aspp_ratios[1],
                        param_attr=param_attr))
        with scope("aspp3"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp3 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[2], act=relu)
            else:
                aspp3 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[2],
                        padding=aspp_ratios[2],
                        param_attr=param_attr))
        with scope("concat"):
            data = fluid.layers.concat([image_avg, aspp0, aspp1, aspp2, aspp3],
                                       axis=1)
            data = bn_relu(
                conv(
                    data,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
            data = fluid.layers.dropout(data, 0.9)
        return data


def decoder(encode_data, decode_shortcut):
    # 解码器配置
    # encode_data：编码器输出
    # decode_shortcut: 从backbone引出的分支, resize后与encode_data concat
    # DECODER_USE_SEP_CONV: 默认为真，则concat后连接两个可分离卷积，否则为普通卷积
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('decoder'):
        with scope('concat'):
            decode_shortcut = bn_relu(
                conv(
                    decode_shortcut,
                    48,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))

            encode_data = fluid.layers.resize_bilinear(
                encode_data, decode_shortcut.shape[2:])
            encode_data = fluid.layers.concat([encode_data, decode_shortcut],
                                              axis=1)
        if cfg.MODEL.DEEPLAB.DECODER_USE_SEP_CONV:
            with scope("separable_conv1"):
                encode_data = separate_conv(
                    encode_data, 256, 1, 3, dilation=1, act=relu)
            with scope("separable_conv2"):
                encode_data = separate_conv(
                    encode_data, 256, 1, 3, dilation=1, act=relu)
        else:
            with scope("decoder_conv1"):
                encode_data = bn_relu(
                    conv(
                        encode_data,
                        256,
                        stride=1,
                        filter_size=3,
                        dilation=1,
                        padding=1,
                        param_attr=param_attr))
            with scope("decoder_conv2"):
                encode_data = bn_relu(
                    conv(
                        encode_data,
                        256,
                        stride=1,
                        filter_size=3,
                        dilation=1,
                        padding=1,
                        param_attr=param_attr))
        return encode_data


def mobilenetv2(input):
    # Backbone: mobilenetv2结构配置
    # DEPTH_MULTIPLIER: mobilenetv2的scale设置，默认1.0
    # OUTPUT_STRIDE：下采样倍数
    # end_points: mobilenetv2的block数
    # decode_point: 从mobilenetv2中引出分支所在block数, 作为decoder输入
    scale = cfg.MODEL.DEEPLAB.DEPTH_MULTIPLIER
    output_stride = cfg.MODEL.DEEPLAB.OUTPUT_STRIDE
    model = mobilenet_backbone(scale=scale, output_stride=output_stride)
    end_points = 18
    decode_point = 4
    data, decode_shortcuts = model.net(
        input, end_points=end_points, decode_points=decode_point)
    decode_shortcut = decode_shortcuts[decode_point]
    return data, decode_shortcut


def xception(input):
    # Backbone: Xception结构配置, xception_65, xception_41, xception_71三种可选
    # decode_point: 从Xception中引出分支所在block数，作为decoder输入
    # end_point：Xception的block数
    cfg.MODEL.DEFAULT_EPSILON = 1e-3
    model = xception_backbone(cfg.MODEL.DEEPLAB.BACKBONE)
    backbone = cfg.MODEL.DEEPLAB.BACKBONE
    output_stride = cfg.MODEL.DEEPLAB.OUTPUT_STRIDE
    if '65' in backbone:
        decode_point = 2
        end_points = 21
    if '41' in backbone:
        decode_point = 2
        end_points = 13
    if '71' in backbone:
        decode_point = 3
        end_points = 23
    data, decode_shortcuts = model.net(
        input,
        output_stride=output_stride,
        end_points=end_points,
        decode_points=decode_point)
    decode_shortcut = decode_shortcuts[decode_point]
    return data, decode_shortcut


def deeplabv3p(img, num_classes):
    # Backbone设置：xception 或 mobilenetv2
    if 'xception' in cfg.MODEL.DEEPLAB.BACKBONE:
        data, decode_shortcut = xception(img)
    elif 'mobilenet' in cfg.MODEL.DEEPLAB.BACKBONE:
        data, decode_shortcut = mobilenetv2(img)
    else:
        raise Exception("deeplab only support xception and mobilenet backbone")

    # 编码器解码器设置
    cfg.MODEL.DEFAULT_EPSILON = 1e-5
    if cfg.MODEL.DEEPLAB.ENCODER_WITH_ASPP:
        data = encoder(data)
    if cfg.MODEL.DEEPLAB.ENABLE_DECODER:
        data = decoder(data, decode_shortcut)

    # 根据类别数设置最后一个卷积层输出，并resize到图片原始尺寸
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    with scope('logit'):
        logit = conv(
            data,
            num_classes,
            1,
            stride=1,
            padding=0,
            bias_attr=True,
            param_attr=param_attr)
        logit = fluid.layers.resize_bilinear(logit, img.shape[2:])

    return logit
