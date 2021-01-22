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
import contextlib
import paddle
import paddle.static as static
import paddle.static.nn as nn
import paddle.nn.functional as F
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, bn_relu, relu, qsigmoid
from models.libs.model_libs import conv
from models.libs.model_libs import separate_conv
from models.backbone.resnet_vd import ResNet as resnet_vd_backbone


def encoder(input):
    # 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
    # ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
    # OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
    # aspp_ratios：ASPP模块空洞卷积的采样率

    if not cfg.MODEL.DEEPLAB.ENCODER.ASPP_RATIOS:
        if cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 16:
            aspp_ratios = [6, 12, 18]
        elif cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 8:
            aspp_ratios = [12, 24, 36]
        else:
            aspp_ratios = []
    else:
        aspp_ratios = cfg.MODEL.DEEPLAB.ENCODER.ASPP_RATIOS

    param_attr = paddle.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.06))

    concat_logits = []
    with scope('encoder'):
        channel = cfg.MODEL.DEEPLAB.ENCODER.ASPP_CONVS_FILTERS
        with scope("image_pool"):
            image_avg = F.adaptive_avg_pool2d(input, output_size=(1, 1))
            act = qsigmoid if cfg.MODEL.DEEPLAB.ENCODER.SE_USE_QSIGMOID else bn_relu
            image_avg = act(
                conv(
                    image_avg,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
            image_avg = F.interpolate(
                image_avg, input.shape[2:], mode='bilinear', align_corners=True)
            if cfg.MODEL.DEEPLAB.ENCODER.ADD_IMAGE_LEVEL_FEATURE:
                concat_logits.append(image_avg)

        with scope("aspp0"):
            aspp0 = bn_relu(
                conv(
                    input,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr,
                    bias_attr=None))
            aspp0 = F.interpolate(
                aspp0, input.shape[2:], mode='bilinear', align_corners=True)
            concat_logits.append(aspp0)

        if aspp_ratios:
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
                aspp1 = F.interpolate(
                    aspp1, input.shape[2:], mode='bilinear', align_corners=True)
                concat_logits.append(aspp1)
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
                aspp2 = F.interpolate(
                    aspp2, input.shape[2:], mode='bilinear', align_corners=True)
                concat_logits.append(aspp2)
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
                aspp3 = F.interpolate(
                    aspp3, input.shape[2:], mode='bilinear', align_corners=True)
                concat_logits.append(aspp3)

        with scope("concat"):
            data = paddle.concat(concat_logits, axis=1)
            if cfg.MODEL.DEEPLAB.ENCODER.ASPP_WITH_CONCAT_PROJECTION:
                data = bn_relu(
                    conv(
                        data,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr,
                        bias_attr=None))
                data = F.dropout(data, 0.1, mode='downscale_in_infer')

        if cfg.MODEL.DEEPLAB.ENCODER.ASPP_WITH_SE:
            data = data * image_avg
        return data


def _decoder_with_sum_merge(encode_data, decode_shortcut, param_attr):
    encode_data = F.interpolate(
        encode_data,
        decode_shortcut.shape[2:],
        mode='bilinear',
        align_corners=True)
    encode_data = conv(
        encode_data,
        cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
        1,
        1,
        groups=1,
        padding=0,
        param_attr=param_attr)

    with scope('merge'):
        decode_shortcut = conv(
            decode_shortcut,
            cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
            1,
            1,
            groups=1,
            padding=0,
            param_attr=param_attr)

        return encode_data + decode_shortcut


def _decoder_with_concat(encode_data, decode_shortcut, param_attr):
    with scope('concat'):
        decode_shortcut = bn_relu(
            conv(
                decode_shortcut,
                48,
                1,
                1,
                groups=1,
                padding=0,
                param_attr=param_attr,
                bias_attr=None))

        encode_data = F.interpolate(
            encode_data,
            decode_shortcut.shape[2:],
            mode='bilinear',
            align_corners=True)
        encode_data = paddle.concat([encode_data, decode_shortcut], axis=1)
    if cfg.MODEL.DEEPLAB.DECODER_USE_SEP_CONV:
        with scope("separable_conv1"):
            encode_data = separate_conv(
                encode_data,
                cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
                1,
                3,
                dilation=1)
        with scope("separable_conv2"):
            encode_data = separate_conv(
                encode_data,
                cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
                1,
                3,
                dilation=1)
    else:
        with scope("decoder_conv1"):
            encode_data = bn_relu(
                conv(
                    encode_data,
                    cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
                    stride=1,
                    filter_size=3,
                    dilation=1,
                    padding=1,
                    param_attr=param_attr))
        with scope("decoder_conv2"):
            encode_data = bn_relu(
                conv(
                    encode_data,
                    cfg.MODEL.DEEPLAB.DECODER.CONV_FILTERS,
                    stride=1,
                    filter_size=3,
                    dilation=1,
                    padding=1,
                    param_attr=param_attr))
    return encode_data


def decoder(encode_data, decode_shortcut):
    # 解码器配置
    # encode_data：编码器输出
    # decode_shortcut: 从backbone引出的分支, resize后与encode_data concat
    # DECODER_USE_SEP_CONV: 默认为真，则concat后连接两个可分离卷积，否则为普通卷积
    param_attr = paddle.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.06))
    with scope('decoder'):
        if cfg.MODEL.DEEPLAB.DECODER.USE_SUM_MERGE:
            return _decoder_with_sum_merge(encode_data, decode_shortcut,
                                           param_attr)

        return _decoder_with_concat(encode_data, decode_shortcut, param_attr)


def resnet_vd(input):
    # backbone: resnet_vd, 可选resnet50_vd, resnet101_vd
    # end_points: resnet终止层数
    # dilation_dict: resnet block数及对应的膨胀卷积尺度
    backbone = cfg.MODEL.DEEPLAB.BACKBONE
    if '50' in backbone:
        layers = 50
    elif '101' in backbone:
        layers = 101
    else:
        raise Exception("resnet_vd backbone only support layers 50 or 101")
    output_stride = cfg.MODEL.DEEPLAB.OUTPUT_STRIDE
    end_points = layers - 1
    decode_point = 10
    if output_stride == 8:
        dilation_dict = {2: 2, 3: 4}
    elif output_stride == 16:
        dilation_dict = {3: 2}
    else:
        raise Exception("deeplab only support stride 8 or 16")
    lr_mult_list = cfg.MODEL.DEEPLAB.BACKBONE_LR_MULT_LIST
    if lr_mult_list is None:
        lr_mult_list = [1.0, 1.0, 1.0, 1.0, 1.0]
    model = resnet_vd_backbone(
        layers, stem='deeplab', lr_mult_list=lr_mult_list)
    data, decode_shortcuts = model.net(
        input,
        end_points=end_points,
        decode_points=decode_point,
        dilation_dict=dilation_dict)
    decode_shortcut = decode_shortcuts[decode_point]

    return data, decode_shortcut


def deeplabv3p(img, num_classes):
    # Backbone设置：xception 或 mobilenetv2
    if 'resnet' in cfg.MODEL.DEEPLAB.BACKBONE:
        data, decode_shortcut = resnet_vd(img)
    else:
        raise Exception("deeplab only support resnet_vd backbone")

    # 编码器解码器设置
    cfg.MODEL.DEFAULT_EPSILON = 1e-5
    if cfg.MODEL.DEEPLAB.ENCODER_WITH_ASPP:
        data = encoder(data)
    if cfg.MODEL.DEEPLAB.ENABLE_DECODER:
        data = decoder(data, decode_shortcut)

    # 根据类别数设置最后一个卷积层输出，并resize到图片原始尺寸
    param_attr = paddle.ParamAttr(
        name=name_scope + 'weights',
        regularizer=paddle.regularizer.L2Decay(coeff=0.0),
        initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.01))

    if not cfg.MODEL.DEEPLAB.DECODER.OUTPUT_IS_LOGITS:
        with scope('logit'):
            with static.name_scope('last_conv'):
                logit = conv(
                    data,
                    num_classes,
                    1,
                    stride=1,
                    padding=0,
                    bias_attr=True,
                    param_attr=param_attr)
    else:
        logit = data

    logit = F.interpolate(
        logit, img.shape[2:], mode='bilinear', align_corners=True)
    return logit
