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
import paddle.fluid as fluid
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, bn_relu, relu
from models.libs.model_libs import conv, max_pool, deconv


def double_conv(data, out_ch):
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.33))
    with scope("conv0"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1, param_attr=param_attr))
    with scope("conv1"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1, param_attr=param_attr))
    return data


def down(data, out_ch):
    # 下采样：max_pool + 2个卷积
    with scope("down"):
        data = max_pool(data, 2, 2, 0)
        data = double_conv(data, out_ch)
    return data


def up(data, short_cut, out_ch):
    # 上采样：data上采样(resize或deconv), 并与short_cut concat
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.XavierInitializer(),
    )
    with scope("up"):
        if cfg.MODEL.UNET.UPSAMPLE_MODE == 'bilinear':
            data = fluid.layers.resize_bilinear(data, short_cut.shape[2:])
        else:
            data = deconv(
                data,
                out_ch // 2,
                filter_size=2,
                stride=2,
                padding=0,
                param_attr=param_attr)
        data = fluid.layers.concat([data, short_cut], axis=1)
        data = double_conv(data, out_ch)
    return data


def encode(data):
    # 编码器设置
    short_cuts = []
    with scope("encode"):
        with scope("block1"):
            data = double_conv(data, 64)
            short_cuts.append(data)
        with scope("block2"):
            data = down(data, 128)
            short_cuts.append(data)
        with scope("block3"):
            data = down(data, 256)
            short_cuts.append(data)
        with scope("block4"):
            data = down(data, 512)
            short_cuts.append(data)
        with scope("block5"):
            data = down(data, 512)
    return data, short_cuts


def decode(data, short_cuts):
    # 解码器设置，与编码器对称
    with scope("decode"):
        with scope("decode1"):
            data = up(data, short_cuts[3], 256)
        with scope("decode2"):
            data = up(data, short_cuts[2], 128)
        with scope("decode3"):
            data = up(data, short_cuts[1], 64)
        with scope("decode4"):
            data = up(data, short_cuts[0], 64)
    return data


def get_logit(data, num_classes):
    # 根据类别数设置最后一个卷积层输出
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    with scope("logit"):
        data = conv(
            data, num_classes, 3, stride=1, padding=1, param_attr=param_attr)
    return data


def unet(input, num_classes):
    # UNET网络配置，对称的编码器解码器
    encode_data, short_cuts = encode(input)
    decode_data = decode(encode_data, short_cuts)
    logit = get_logit(decode_data, num_classes)
    return logit


if __name__ == '__main__':
    image_shape = [-1, 3, 320, 320]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    logit = unet(image, 4)
    print("logit:", logit.shape)
