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
from utils.config import cfg
from models.libs.model_libs import scope
from models.libs.model_libs import bn, avg_pool, conv
from models.backbone.resnet import ResNet as resnet_backbone
import numpy as np


def interp(input, out_shape):
    out_shape = list(out_shape.astype("int32"))
    return fluid.layers.resize_bilinear(input, out_shape=out_shape)


def pyramis_pooling(input, input_shape):
    shape = np.ceil(input_shape / 32).astype("int32")
    h, w = shape
    pool1 = avg_pool(input, [h, w], [h, w])
    pool1_interp = interp(pool1, shape)
    pool2 = avg_pool(input, [h // 2, w // 2], [h // 2, w // 2])
    pool3 = avg_pool(input, [h // 3, w // 3], [h // 3, w // 3])
    pool4 = avg_pool(input, [h // 4, w // 4], [h // 4, w // 4])
    # official caffe repo eval use following hyparam
    # pool2 = avg_pool(input, [17, 33], [16, 32])
    # pool3 = avg_pool(input, [13, 25], [10, 20])
    # pool4 = avg_pool(input, [8, 15], [5, 10])
    pool2_interp = interp(pool2, shape)
    pool3_interp = interp(pool3, shape)
    pool4_interp = interp(pool4, shape)
    conv5_3_sum = input + pool4_interp + pool3_interp + pool2_interp + pool1_interp
    return conv5_3_sum


def zero_padding(input, padding):
    return fluid.layers.pad(input,
                            [0, 0, 0, 0, padding, padding, padding, padding])


def sub_net_4(input, input_shape):
    tmp = pyramis_pooling(input, input_shape)
    with scope("conv5_4_k1"):
        tmp = conv(tmp, 256, 1, 1)
        tmp = bn(tmp, act='relu')
    tmp = interp(tmp, out_shape=np.ceil(input_shape / 16))
    return tmp


def sub_net_2(input):
    with scope("conv3_1_sub2_proj"):
        tmp = conv(input, 128, 1, 1)
        tmp = bn(tmp)
    return tmp


def sub_net_1(input):
    with scope("conv1_sub1"):
        tmp = conv(input, 32, 3, 2, padding=1)
        tmp = bn(tmp, act='relu')
    with scope("conv2_sub1"):
        tmp = conv(tmp, 32, 3, 2, padding=1)
        tmp = bn(tmp, act='relu')
    with scope("conv3_sub1"):
        tmp = conv(tmp, 64, 3, 2, padding=1)
        tmp = bn(tmp, act='relu')
    with scope("conv3_sub1_proj"):
        tmp = conv(tmp, 128, 1, 1)
        tmp = bn(tmp)
    return tmp


def CCF24(sub2_out, sub4_out, input_shape):
    with scope("conv_sub4"):
        tmp = conv(sub4_out, 128, 3, dilation=2, padding=2)
        tmp = bn(tmp)
    tmp = tmp + sub2_out
    tmp = fluid.layers.relu(tmp)
    tmp = interp(tmp, np.ceil(input_shape / 8))
    return tmp


def CCF124(sub1_out, sub24_out, input_shape):
    tmp = zero_padding(sub24_out, padding=2)
    with scope("conv_sub2"):
        tmp = conv(tmp, 128, 3, dilation=2)
        tmp = bn(tmp)
    tmp = tmp + sub1_out
    tmp = fluid.layers.relu(tmp)
    tmp = interp(tmp, input_shape // 4)
    return tmp


def resnet(input):
    # ICNET backbone: resnet, 默认resnet50
    # end_points: resnet终止层数
    # decode_point: backbone引出分支所在层数
    # resize_point：backbone所在的该层卷积尺寸缩小至1/2
    # dilation_dict: resnet block数及对应的膨胀卷积尺度
    scale = cfg.MODEL.ICNET.DEPTH_MULTIPLIER
    layers = cfg.MODEL.ICNET.LAYERS
    model = resnet_backbone(scale=scale, layers=layers, stem='icnet')
    end_points = 49
    decode_point = 13
    resize_point = 13
    dilation_dict = {2: 2, 3: 4}
    data, decode_shortcuts = model.net(
        input,
        end_points=end_points,
        decode_points=decode_point,
        resize_points=resize_point,
        dilation_dict=dilation_dict)
    return data, decode_shortcuts[decode_point]


def encoder(data13, data49, input, input_shape):
    # ICENT encoder配置
    # sub_net_4：对resnet49层数据进行pyramis_pooling操作
    # sub_net_2：对resnet13层数据进行卷积操作
    # sub_net_1: 对原始尺寸图像进行3次下采样卷积操作
    sub4_out = sub_net_4(data49, input_shape)
    sub2_out = sub_net_2(data13)
    sub1_out = sub_net_1(input)
    return sub1_out, sub2_out, sub4_out


def decoder(sub1_out, sub2_out, sub4_out, input_shape):
    # ICENT decoder配置
    # CCF: Cascade Feature Fusion 级联特征融合
    sub24_out = CCF24(sub2_out, sub4_out, input_shape)
    sub124_out = CCF124(sub1_out, sub24_out, input_shape)
    return sub24_out, sub124_out


def get_logit(data, num_classes, name="logit"):
    param_attr = fluid.ParamAttr(
        name=name + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))

    with scope(name):
        data = conv(
            data,
            num_classes,
            1,
            stride=1,
            padding=0,
            param_attr=param_attr,
            bias_attr=True)
    return data


def icnet(input, num_classes):
    # Backbone resnet: 输入 image_sub2: 图片尺寸缩小至1/2
    #                  输出 data49: resnet第49层数据，原始尺寸1/32
    #                       data13：resnet第13层数据, 原始尺寸1/16
    input_shape = input.shape[2:]
    input_shape = np.array(input_shape).astype("float32")
    image_sub2 = interp(input, out_shape=np.ceil(input_shape * 0.5))
    data49, data13 = resnet(image_sub2)

    # encoder：输入：input, data13, data49，分别进行下采样，卷积和金字塔pooling操作
    #          输出：分别对应sub1_out, sub2_out, sub4_out
    sub1_out, sub2_out, sub4_out = encoder(data13, data49, input, input_shape)

    # decoder: 对编码器三个分支结果进行级联特征融合
    sub24_out, sub124_out = decoder(sub1_out, sub2_out, sub4_out, input_shape)

    # get_logit: 根据类别数决定最后一层卷积输出
    logit124 = get_logit(sub124_out, num_classes, "logit124")
    logit4 = get_logit(sub4_out, num_classes, "logit4")
    logit24 = get_logit(sub24_out, num_classes, "logit24")
    return logit124, logit24, logit4


if __name__ == '__main__':
    image_shape = [-1, 3, 320, 320]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    logit = icnet(image, 4)
    print("logit:", logit.shape)
