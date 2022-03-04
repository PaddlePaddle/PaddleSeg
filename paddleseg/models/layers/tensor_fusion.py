# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers


def get_pure_tensor(x):
    if not isinstance(x, (list, tuple)):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        return paddle.concat(x, axis=1)


def avg_reduce_hw(x):
    # Reduce hw by avg
    # Return cat([avg_pool_0, avg_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return paddle.concat(res, axis=1)


def avg_reduce_hw_slow(x):
    # Reduce hw by avg, equal to avg_reduce_hw
    x = get_pure_tensor(x)
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    return avg_pool


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when axis=[2, 3], the paddle.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = paddle.max(x, axis=[2, 3], keepdim=True)

    if use_concat:
        res = paddle.concat([avg_pool, max_pool], axis=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return paddle.concat(res, axis=1)


def avg_max_reduce_hw_slow(x, is_training):
    # Reduce hw by avg and max, equal to avg_max_reduce_hw
    x = get_pure_tensor(x)

    avg_pool = F.adaptive_avg_pool2d(x, 1)
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = paddle.max(x, axis=[2, 3], keepdim=True)

    res = paddle.concat([avg_pool, max_pool], axis=1)
    return res


def raw_avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return [cat([avg_pool_0, avg_pool_1, ...], cat([max_pool_0, max_pool_1, ...]))]
    assert isinstance(x, (list, tuple)) and len(x) > 1

    res_avg = []
    res_max = []
    for xi in x:
        avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
        res_avg.append(avg)
        res_max.append(max)
    res_avg = paddle.concat(res_avg, axis=1)
    res_max = paddle.concat(res_max, axis=1)

    return res_avg, res_max


def avg_reduce_channel(x):
    # Reduce channel by avg
    # Return cat([avg_ch_0, avg_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return paddle.mean(x, axis=1, keepdim=True)
    elif len(x) == 1:
        return paddle.mean(x[0], axis=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(paddle.mean(xi, axis=1, keepdim=True))
        return paddle.concat(res, axis=1)


def max_reduce_channel(x):
    # Reduce channel by max
    # Return cat([max_ch_0, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return paddle.max(x, axis=1, keepdim=True)
    elif len(x) == 1:
        return paddle.max(x[0], axis=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(paddle.max(xi, axis=1, keepdim=True))
        return paddle.concat(res, axis=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = paddle.mean(x, axis=1, keepdim=True)
    max_value = paddle.max(x, axis=1, keepdim=True)

    if use_concat:
        res = paddle.concat([mean_value, max_value], axis=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return paddle.concat(res, axis=1)


def avg_max_reduce_channel_1(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, avg_ch_1, ..., max_ch_0, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_channel_helper(xi, False)
            res_avg.append(avg)
            res_max.append(max)

        res = res_avg + res_max
        return paddle.concat(res, axis=1)


def raw_avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, avg_ch_1, ...]), cat([max_ch_0, max_ch_1, ...])
    assert isinstance(x, (list, tuple)) and len(x) > 1

    res_avg = []
    res_max = []
    for xi in x:
        avg, max = avg_max_reduce_channel_helper(xi, False)
        res_avg.append(avg)
        res_max.append(max)

    res_avg = paddle.concat(res_avg, axis=1)
    res_max = paddle.concat(res_max, axis=1)

    return res_avg, res_max


def cat_avg_max_reduce_channel(x):
    # Reduce hw by cat+avg+max
    assert isinstance(x, (list, tuple)) and len(x) > 1

    x = paddle.concat(x, axis=1)

    mean_value = paddle.mean(x, axis=1, keepdim=True)
    max_value = paddle.max(x, axis=1, keepdim=True)
    res = paddle.concat([mean_value, max_value], axis=1)

    return res


def combine_avg_max_reduce_channel(x):
    # Reduce hw by cat+avg+max, equal to cat_avg_max_reduce_channel
    assert isinstance(x, (list, tuple)) and len(x) > 1

    res_avg = []
    res_max = []
    for xi in x:
        avg, max = avg_max_reduce_channel_helper(xi, False)
        res_avg.append(avg)
        res_max.append(max)

    avg = paddle.mean(paddle.concat(res_avg, axis=1), axis=1, keepdim=True)
    max = paddle.max(paddle.concat(res_max, axis=1), axis=1, keepdim=True)
    res = paddle.concat([avg, max], axis=1)

    return res


class ARM_Add_Add(nn.Layer):
    """
    Fuse two tensors. xs are several bigger tensors, y is smaller tensor.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        if isinstance(x_chs, int):
            self.x_num = 1
            x_ch = x_chs
        elif len(x_chs) == 1:
            self.x_num = 1
            x_ch = x_chs[0]
        else:
            self.x_num = len(x_chs)
            x_ch = x_chs[0]
            assert all([x_ch == ch for ch in x_chs]), \
                "All value in x_chs should be equal"

        self.conv_x = layers.ConvBNReLU(x_ch,
                                        y_ch,
                                        kernel_size=ksize,
                                        padding=ksize // 2,
                                        bias_attr=False)
        self.conv_out = layers.ConvBNReLU(y_ch,
                                          out_ch,
                                          kernel_size=3,
                                          padding=1,
                                          bias_attr=False)
        self.resize_mode = resize_mode

    def check(self, xs, y):
        # check num
        x_num = 1 if not isinstance(xs, (list, tuple)) else len(xs)
        assert x_num == self.x_num, \
            "The nums of xs ({}) should be equal to {}".format(x_num, self.x_num)

        # check shape
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, xs, y):
        x = self.prepare_x(xs, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x += xs[i]

        x = self.conv_x(x)
        return x
        '''
        # (TODO)use sum will add scale op to infer model, check the speed by trt
        if not isinstance(xs, (list, tuple)):
            x = xs
        else:
            x = sum(xs)
        x = self.conv_x(x)
        return x
        '''

    def prepare_y(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[-1]
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, xs, y):
        self.check(xs, y)
        x, y = self.prepare(xs, y)
        out = self.fuse(x, y)
        return out


class ARM_WeightedAdd0_Add(ARM_Add_Add):
    """Add two tensor"""
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        alpha = self.create_parameter([self.x_num])
        self.add_parameter("alpha", alpha)

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha[0] * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.alpha[i] * xs[i]

        x = self.conv_x(x)
        return x


class ARM_WeightedAdd1_Add(ARM_Add_Add):
    """Add two tensor"""
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert self.x_num == 2, "ARM_WeightedAdd1_Add requires x_num = 2"

        alpha = self.create_parameter([1])
        self.add_parameter("alpha", alpha)

    def prepare_x(self, xs, y):
        assert isinstance(xs, (list, tuple)) and len(xs) == 2

        w = F.sigmoid(self.alpha[0])
        x = xs[0] * w + xs[1] * (1 - w)

        x = self.conv_x(x)
        return x


class ARM_WeightedAdd2_Add(ARM_Add_Add):
    """Add two tensor"""
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        for i, ch in enumerate(x_chs):
            name = "alpha_" + str(i)
            self.add_parameter(name, self.create_parameter([ch, 1, 1]))

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha_0 * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                alpha_name = "alpha_" + str(i)
                alpha = getattr(self, alpha_name)
                x = x + alpha * xs[i]

        x = self.conv_x(x)
        return x


class ARM_SEAdd1_Add(ARM_Add_Add):
    """Add two tensor"""
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def sp_atten(self, input):
        if self.training:
            avg_pool = F.adaptive_avg_pool2d(input, 1)
        else:
            avg_pool = paddle.mean(input, axis=[2, 3], keepdim=True)
        atten = F.sigmoid(avg_pool)
        return atten * input

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.sp_atten(x)

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.sp_atten(xs[i])

        x = self.conv_x(x)
        return x


class ARM_ChAttenAdd0_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(sum(x_chs),
                                           x_chs[0],
                                           kernel_size=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_reduce_hw(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_ChAttenAdd1_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(2 * sum(x_chs),
                                           x_chs[0],
                                           kernel_size=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_max_reduce_hw(xs, self.training)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_ChAttenAdd2_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = nn.Sequential(
            layers.ConvBNAct(sum(x_chs),
                             x_chs[0] // 2,
                             kernel_size=1,
                             bias_attr=False,
                             act_type="leakyrelu"),
            layers.ConvBN(x_chs[0] // 2,
                          x_chs[0],
                          kernel_size=1,
                          bias_attr=False))

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_reduce_hw(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_ChAttenAdd3_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = nn.Sequential(
            layers.ConvBNAct(2 * sum(x_chs),
                             x_chs[0] // 2,
                             kernel_size=1,
                             bias_attr=False,
                             act_type="leakyrelu"),
            layers.ConvBN(x_chs[0] // 2,
                          x_chs[0],
                          kernel_size=1,
                          bias_attr=False))

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_max_reduce_hw(xs, self.training)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_SpAttenAdd0_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_reduce_channel(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_SpAttenAdd1_Add(ARM_Add_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert isinstance(x_chs, (list, tuple)) and len(x_chs) == 2, \
            "x_chs should be (list, tuple) and the length should be 2"

        self.conv_xs_atten = layers.ConvBN(4,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        # xs is [x1, x2]
        atten = avg_max_reduce_channel(xs)
        atten = F.sigmoid(self.conv_xs_atten(atten))

        x = xs[0] * atten + xs[1] * (1 - atten)
        x = self.conv_x(x)
        return x


class ARM_Add_ChAttenAdd0(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2 * y_ch,
                                           y_ch,
                                           kernel_size=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_ChAttenAdd1(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(4 * y_ch,
                                           y_ch,
                                           kernel_size=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_ChAttenAdd2(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(2 * y_ch,
                             y_ch // 2,
                             kernel_size=1,
                             bias_attr=False,
                             act_type="leakyrelu"),
            layers.ConvBN(y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_ChAttenAdd3(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(4 * y_ch,
                             y_ch // 2,
                             kernel_size=1,
                             bias_attr=False,
                             act_type="leakyrelu"),
            layers.ConvBN(y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd0(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd0_1(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * (0.5 + atten) + y * (1.5 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd1(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(4,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd2(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(2, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd2_1(ARM_Add_Add):
    """
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(2, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * (0.5 + atten) + y * (1.5 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd3(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd4(ARM_Add_Add):
    """
    use cat_avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = cat_avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd4_1(ARM_Add_Add):
    """
    use cat_avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = cat_avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * (0.5 + atten) + y * (1.5 - atten)
        out = self.conv_out(out)
        return out


class ARM_Add_SpAttenAdd5(ARM_Add_Add):
    """
    use cat_avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(2, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        atten = cat_avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_WeightedAdd0_SpAttenAdd1(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        alpha = self.create_parameter([self.x_num])
        self.add_parameter("alpha", alpha)

        self.conv_xy_atten = layers.ConvBN(4,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha[0] * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.alpha[i] * xs[i]

        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_WeightedAdd0_SpAttenAdd4(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        alpha = self.create_parameter([self.x_num])
        self.add_parameter("alpha", alpha)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha[0] * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                x = x + self.alpha[i] * xs[i]

        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        atten = cat_avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_WeightedAdd1_SpAttenAdd1(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert self.x_num == 2, "ARM_WeightedAdd1_Add requires x_num = 2"

        alpha = self.create_parameter([1])
        self.add_parameter("alpha", alpha)

        self.conv_xy_atten = layers.ConvBN(4,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def prepare_x(self, xs, y):
        assert isinstance(xs, (list, tuple)) and len(xs) == 2

        w = F.sigmoid(self.alpha[0])
        x = xs[0] * w + xs[1] * (1 - w)

        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_WeightedAdd2_SpAttenAdd1(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert self.x_num == 2, "ARM_WeightedAdd1_Add requires x_num = 2"

        self.conv_xy_atten = layers.ConvBN(4,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

        for i, ch in enumerate(x_chs):
            name = "alpha_" + str(i)
            self.add_parameter(name, self.create_parameter([ch, 1, 1]))

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha_0 * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                alpha_name = "alpha_" + str(i)
                alpha = getattr(self, alpha_name)
                x = x + alpha * xs[i]

        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_WeightedAdd2_SpAttenAdd4(ARM_Add_Add):
    """
    use avg_max_reduce_channel
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        assert self.x_num == 2, "ARM_WeightedAdd1_Add requires x_num = 2"

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

        for i, ch in enumerate(x_chs):
            name = "alpha_" + str(i)
            self.add_parameter(name, self.create_parameter([ch, 1, 1]))

    def prepare_x(self, xs, y):
        x = xs if not isinstance(xs, (list, tuple)) else xs[0]
        x = self.alpha_0 * x

        if self.x_num > 1:
            for i in range(1, self.x_num):
                alpha_name = "alpha_" + str(i)
                alpha = getattr(self, alpha_name)
                x = x + alpha * xs[i]

        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        atten = cat_avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_ChAttenAdd0_ChAttenAdd0(ARM_ChAttenAdd0_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2 * y_ch,
                                           y_ch,
                                           kernel_size=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_ChAttenAdd0_SpAttenAdd0(ARM_ChAttenAdd0_Add):
    """
    The length of x_chs and xs should be 2.
    """
    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = layers.ConvBN(2,
                                           1,
                                           kernel_size=3,
                                           padding=1,
                                           bias_attr=False)

    def fuse(self, x, y):
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out
