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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    'MobileNetV3', 'MobileNetV3_small_x0_35', 'MobileNetV3_small_x0_5',
    'MobileNetV3_small_x0_75', 'MobileNetV3_small_x1_0',
    'MobileNetV3_small_x1_25', 'MobileNetV3_large_x0_35',
    'MobileNetV3_large_x0_5', 'MobileNetV3_large_x0_75',
    'MobileNetV3_large_x1_0', 'MobileNetV3_large_x1_25'
]


class MobileNetV3():
    def __init__(self,
                 scale=1.0,
                 model_name='small',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 output_stride=None):
        self.scale = scale
        self.inplanes = 16

        self.lr_mult_list = lr_mult_list
        assert len(self.lr_mult_list) == 5, \
            "lr_mult_list length in MobileNetV3 must be 5 but got {}!!".format(
            len(self.lr_mult_list))
        self.curr_stage = 0
        self.decode_point = None
        self.end_point = None

        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                # The number of channels in the last 4 stages is reduced by a
                # factor of 2 compared to the standard implementation.
                [5, 336, 80, True, 'hard_swish', 2],
                [5, 480, 80, True, 'hard_swish', 1],
                [5, 480, 80, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 480
            self.cls_ch_expand = 1280
            self.lr_interval = 3
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                # The number of channels in the last 4 stages is reduced by a
                # factor of 2 compared to the standard implementation.
                [5, 144, 48, True, 'hard_swish', 2],
                [5, 288, 48, True, 'hard_swish', 1],
                [5, 288, 48, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 288
            self.cls_ch_expand = 1280
            self.lr_interval = 2
        else:
            raise NotImplementedError(
                "mode[{}_model] is not implemented!".format(model_name))

        self.modify_bottle_params(output_stride)

    def modify_bottle_params(self, output_stride=None):
        if output_stride is not None and output_stride % 2 != 0:
            raise Exception("output stride must to be even number")
        if output_stride is None:
            return
        else:
            stride = 2
            for i, _cfg in enumerate(self.cfg):
                stride = stride * _cfg[-1]
                if stride > output_stride:
                    s = 1
                    self.cfg[i][-1] = s

    def net(self, input, class_dim=1000, end_points=None, decode_points=None):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand

        # conv1
        conv = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self.make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')

        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        for layer_cfg in cfg:
            conv = self.residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
            self.curr_stage = i

        conv = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=self.make_divisible(scale * cls_ch_squeeze),
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv_last')

        return conv, self.decode_point

        conv = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv = fluid.layers.conv2d(
            input=conv,
            num_filters=cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name='last_1x1_conv_weights'),
            bias_attr=False)
        conv = fluid.layers.hard_swish(conv)
        drop = fluid.layers.dropout(x=conv, dropout_prob=0.2)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            param_attr=ParamAttr(name='fc_weights'),
            bias_attr=ParamAttr(name='fc_offset'))
        return out

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      act=None,
                      name=None,
                      use_cudnn=True,
                      res_last_bn_init=False):
        lr_idx = self.curr_stage // self.lr_interval
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]

        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights', learning_rate=lr_mult),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = fluid.layers.hard_swish(bn)
        return bn

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        lr_idx = self.curr_stage // self.lr_interval
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]

        num_mid_filter = num_out_filter // ratio
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(
                name=name + '_1_weights', learning_rate=lr_mult),
            bias_attr=ParamAttr(name=name + '_1_offset', learning_rate=lr_mult))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(
                name=name + '_2_weights', learning_rate=lr_mult),
            bias_attr=ParamAttr(name=name + '_2_offset', learning_rate=lr_mult))
        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_mid_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      act=None,
                      use_se=False,
                      name=None):

        conv0 = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')

        conv1 = self.conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')

        if self.curr_stage == 5:
            self.decode_point = conv1
        if use_se:
            conv1 = self.se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self.conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear',
            res_last_bn_init=True)
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input, y=conv2, act=None)


def MobileNetV3_small_x0_35():
    model = MobileNetV3(model_name='small', scale=0.35)
    return model


def MobileNetV3_small_x0_5():
    model = MobileNetV3(model_name='small', scale=0.5)
    return model


def MobileNetV3_small_x0_75():
    model = MobileNetV3(model_name='small', scale=0.75)
    return model


def MobileNetV3_small_x1_0(**args):
    model = MobileNetV3(model_name='small', scale=1.0, **args)
    return model


def MobileNetV3_small_x1_25():
    model = MobileNetV3(model_name='small', scale=1.25)
    return model


def MobileNetV3_large_x0_35():
    model = MobileNetV3(model_name='large', scale=0.35)
    return model


def MobileNetV3_large_x0_5():
    model = MobileNetV3(model_name='large', scale=0.5)
    return model


def MobileNetV3_large_x0_75():
    model = MobileNetV3(model_name='large', scale=0.75)
    return model


def MobileNetV3_large_x1_0(**args):
    model = MobileNetV3(model_name='large', scale=1.0, **args)
    return model


def MobileNetV3_large_x1_25():
    model = MobileNetV3(model_name='large', scale=1.25)
    return model
