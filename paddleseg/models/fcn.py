# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


@manager.MODELS.add_component
class FCN(nn.Layer):
    """
    A simple implementation for FCN based on PaddlePaddle.

    The original article refers to
    Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1411.4038).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 channels=None,
                 align_corners=False,
                 pretrained=None,
                 bias=True,
                 data_format="NCHW"):
        super(FCN, self).__init__()

        if data_format != 'NCHW':
            raise ('fcn only support NCHW data format')
        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            channels,
            bias=bias)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class FCNHead(nn.Layer):
    """
    A simple implementation for FCNHead based on PaddlePaddle

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        pretrained (str, optional): The path of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 bias=True):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = layers.ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        self.cls = nn.Conv2D(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegHead, self).__init__()
        self.conv = layers.ConvBNReLU(
            in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


@manager.MODELS.add_component
class FCN_8s(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, -2, -3),
                 arm_type='ARM_0',
                 fpn_ch=128,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        bk_chs = [backbone.feat_channels[i] for i in backbone_indices]
        print(bk_chs)

        self.conv_f32 = layers.ConvBNReLU(
            bk_chs[0], fpn_ch, kernel_size=1, bias_attr=False)

        print("arm type: " + arm_type)
        arm_layer = eval(arm_type)
        self.arm_16 = arm_layer(bk_chs[1], fpn_ch, fpn_ch)
        self.arm_8 = arm_layer(bk_chs[2], fpn_ch, fpn_ch)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        x_8, x_16, x_32 = feat_list[-3], feat_list[-2], feat_list[-1]

        x_32_rf = self.conv_f32(x_32)
        x_16_rf = self.arm_16(x_16, x_32_rf)
        x_8_rf = self.arm_8(x_8, x_16_rf)

        logit_list = [x_8_rf]
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class ARM_0(nn.Layer):
    '''add, x is bigger feature map, y is smaller feature map'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        assert y_chan == out_chan
        self.conv_x = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            padding=first_ksize // 2,
            bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            out_chan, out_chan, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
        out = x + y_up
        out = self.conv_out(out)
        return out


class ARM_1(ARM_0):
    '''cat+conv, x is bigger feature map, y is smaller feature map'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__(x_chan, y_chan, out_chan, first_ksize)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
        xy = paddle.concat([x, y_up], axis=1)
        out = self.conv_out(xy)
        return out


class ARM_7_3(ARM_0):
    '''combined channel attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__(x_chan, y_chan, out_chan, first_ksize)
        self.conv_atten = layers.ConvBN(
            2 * (out_chan + y_chan), out_chan, kernel_size=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
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


class ARM_12_4(ARM_0):
    '''combined spatial attention '''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__(x_chan, y_chan, out_chan, first_ksize)
        self.conv_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
        xy = paddle.concat([x, y_up], axis=1)

        xy_mean = paddle.mean(xy, axis=1, keepdim=True)
        xy_max = paddle.max(xy, axis=1, keepdim=True)
        atten = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(atten))  # n * 1 * h * w

        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_17_1(ARM_0):
    '''one sigmoid to combined ch and sp attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__(x_chan, y_chan, out_chan, first_ksize)

        self.ch_atten = layers.ConvBN(
            2 * (out_chan + y_chan), out_chan, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
        xy = paddle.concat([x, y_up], axis=1)

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

        atten = F.sigmoid(sp_atten * ch_atten)
        out = x * atten + y_up * (1 - atten)
        out = self.conv_out(out)
        return out


class ARM_18_1(ARM_0):
    '''one sigmoid to combined ch and sp attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__(x_chan, y_chan, out_chan, first_ksize)

        self.ch_atten = layers.ConvBN(
            2 * (out_chan + y_chan), out_chan, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv_x(x)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode='bilinear')
        xy = paddle.concat([x, y_up], axis=1)

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

        out = ch_atten * x + (1 - ch_atten) * y_up + sp_atten * x + (
            1 - sp_atten) * y_up
        out = self.conv_out(out)
        return out
