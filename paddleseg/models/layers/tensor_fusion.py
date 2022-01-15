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


def check_shape(x, y):
    assert x.ndim == 4 and y.ndim == 4
    x_h, x_w = x.shape[2:]
    y_h, y_w = y.shape[2:]
    assert x_h >= y_h and x_w >= y_w


class FusionBase(nn.Layer):
    """Fuse two tensors. x is bigger tensor, y is smaller tensor."""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def forward(self, x, y):
        pass


class FusionAdd(FusionBase):
    """Add two tensor"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionCat(FusionBase):
    """Concat two tensor"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_out = layers.ConvBNReLU(
            2 * y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)
        out = self.conv_out(xy)
        return out


class FusionChAtten(FusionBase):
    """Use Channel attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
        if self.training:
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        atten = paddle.concat([xy_avg_pool, xy_max_pool], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionSpAtten(FusionBase):
    """Use spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten(FusionBase):
    """Combine channel and spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.ch_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def get_atten(self, xy):
        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_avg_pool = paddle.mean(xy, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        return (ch_atten, sp_atten)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten * ch_atten)
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_1(FusionChSpAtten):
    """atten = F.sigmoid(self.conv_atten(sp_atten * ch_atten))"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(self.conv_atten(sp_atten * ch_atten))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_2(FusionChSpAtten):
    """out = ch_atten * x + (1 - ch_atten) * y_up + sp_atten * x + (
            1 - sp_atten) * y_up"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)
        ch_atten = F.sigmoid(ch_atten)
        sp_atten = F.sigmoid(sp_atten)

        out = ch_atten * x + (1 - ch_atten) * y_up + sp_atten * x + (
            1 - sp_atten) * y_up
        out = self.conv_out(out)
        return out


class FusionChSpAtten_3(FusionChSpAtten):
    """atten = F.sigmoid(sp_atten + ch_atten)"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten + ch_atten)
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionChSpAtten_4(FusionChSpAtten):
    """atten = F.sigmoid(self.conv_atten(sp_atten + ch_atten))"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(self.conv_atten(sp_atten + ch_atten))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionConvAtten(FusionBase):
    """Obtain W by two conv"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * y_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBN(
                y_ch, y_ch, kernel_size=3, padding=1, bias_attr=False))

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        atten = F.sigmoid(self.conv_atten(xy))
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionSF(FusionBase):
    """The fusion in SFNet"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.down_x = nn.Conv2D(y_ch, y_ch // 2, 1, bias_attr=False)
        self.down_y = nn.Conv2D(y_ch, y_ch // 2, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            y_ch, 2, kernel_size=3, padding=1, bias_attr=False)

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)
        norm = size[::-1].reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([size[0]]).transpose([1, 0])
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        x_size = paddle.shape(x)[2:]

        x_flow = self.down_x(x)
        y_flow = self.down_y(y)
        y_flow = F.interpolate(
            y_flow, size=x_size, mode=self.resize_mode, align_corners=False)
        flow = self.flow_make(paddle.concat([x_flow, y_flow], 1))
        y_refine = self.flow_warp(y, flow, size=x_size)

        out = x + y_refine
        out = self.conv_out(out)
        return out


class FusionSFChSpAtten(FusionSF):
    """The fusion in SFNet + combine channel and spatial attention"""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.ch_atten = layers.ConvBN(
            4 * y_ch, y_ch, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def get_atten(self, xy):
        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy, 1)
        else:
            xy_avg_pool = paddle.mean(xy, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        return (ch_atten, sp_atten)

    def forward(self, x, y):
        check_shape(x, y)

        x = self.conv_x(x)
        x_size = paddle.shape(x)[2:]

        x_flow = self.down_x(x)
        y_flow = self.down_y(y)
        y_flow = F.interpolate(
            y_flow, size=x_size, mode=self.resize_mode, align_corners=False)
        flow = self.flow_make(paddle.concat([x_flow, y_flow], 1))
        y_refine = self.flow_warp(y, flow, size=x_size)

        xy = paddle.concat([x, y_refine], axis=1)
        ch_atten, sp_atten = self.get_atten(xy)

        atten = F.sigmoid(sp_atten * ch_atten)
        out = x * atten + y_refine * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionBaseV1(nn.Layer):
    """Fuse two tensors. x is bigger tensor, y is smaller tensor."""

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
            x_num = len(x_chs)
            x_ch = x_chs[0]
            assert all([x_ch == ch for ch in x_chs]), \
                "All value in x_chs should be equal"
            assert x_ch % x_num == 0, \
                "x_ch ({}) should be the multiple of x_num ({})".format(x_ch, x_num)

            self.x_reduction = nn.LayerList()
            for _ in range(len(x_chs)):
                reduction = layers.ConvBNReLU(
                    x_ch, x_ch // x_num, kernel_size=1, bias_attr=False)
                self.x_reduction.append(reduction)

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def forward(self, xs, y):
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

        # x reduction
        if x_num == 1:
            return x
        else:
            x_list = []
            for i in range(x_num):
                x_list.append(self.x_reduction[i](xs[i]))
            x = paddle.concat(x_list, axis=1)
            return x


class FusionAddV1(FusionBaseV1):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def forward(self, xs, y):
        x = super(FusionAddV1, self).forward(xs, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        out = x + y_up
        out = self.conv_out(out)
        return out


class FusionSpAttenV1(FusionBaseV1):
    """Use spatial attention"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, xs, y):
        x = super(FusionSpAttenV1, self).forward(xs, y)

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class FusionBaseV2(nn.Layer):
    """Fuse two tensors. x is bigger tensor, y is smaller tensor."""

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

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def forward(self, xs, y):
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

        # x reduction
        if x_num > 1:
            for i in range(1, x_num):
                x += xs[i]

        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)

        return x, y_up


class FusionAddV2(FusionBaseV2):
    """Add two tensor"""

    def __init__(self, x_chs, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_chs, y_ch, out_ch, ksize, resize_mode)

    def forward(self, xs, y):
        x, y_up = super(FusionAddV2, self).forward(xs, y)

        out = x + y_up
        out = self.conv_out(out)
        return out
