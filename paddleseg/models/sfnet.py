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
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class SFNet(nn.Layer):
    """
    The SFNet implementation based on PaddlePaddle.

    The original article refers to
    Li, Xiangtai, et al. "Semantic Flow for Fast and Accurate Scene Parsing"
    (https://arxiv.org/pdf/2002.10120.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 align_type='AlignedModule_origin',
                 enable_auxiliary_loss=False,
                 align_corners=False,
                 pretrained=None):
        super(SFNet, self).__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss
        if self.backbone.layers == 18:
            fpn_dim = 128
            inplane_head = 512
            fpn_inplanes = [64, 128, 256, 512]
        else:
            fpn_dim = 256
            inplane_head = 2048
            fpn_inplanes = [256, 512, 1024, 2048]

        self.head = SFNetHead(
            inplane=inplane_head,
            num_class=num_classes,
            fpn_inplanes=fpn_inplanes,
            fpn_dim=fpn_dim,
            align_type=align_type,
            enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class SFNetHead(nn.Layer):
    """
    The SFNetHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        fpn_inplanes (list): The feature channels from backbone.
        fpn_dim (int, optional): The input channels of FAM module. Default: 256.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """

    def __init__(self,
                 inplane,
                 num_class,
                 fpn_inplanes,
                 fpn_dim=256,
                 align_type='AlignedModule_origin',
                 enable_auxiliary_loss=False):
        super(SFNetHead, self).__init__()
        self.ppm = layers.PPModule(
            in_channels=inplane,
            out_channels=fpn_dim,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=True,
            align_corners=True)
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.fpn_in = []

        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2D(fpn_inplane, fpn_dim, 1),
                    layers.SyncBatchNorm(fpn_dim), nn.ReLU()))

        self.fpn_in = nn.LayerList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        align_model = eval(align_type)
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                nn.Sequential(
                    layers.ConvBNReLU(fpn_dim, fpn_dim, 3, bias_attr=False)))
            self.fpn_out_align.append(
                align_model(inplane=fpn_dim, outplane=fpn_dim // 2))
            if self.enable_auxiliary_loss:
                self.dsn.append(
                    nn.Sequential(layers.AuxLayer(fpn_dim, fpn_dim, num_class)))

        self.fpn_out = nn.LayerList(self.fpn_out)
        self.fpn_out_align = nn.LayerList(self.fpn_out_align)

        if self.enable_auxiliary_loss:
            self.dsn = nn.LayerList(self.dsn)

        self.conv_last = nn.Sequential(
            layers.ConvBNReLU(
                len(fpn_inplanes) * fpn_dim, fpn_dim, 3, bias_attr=False),
            nn.Conv2D(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            #f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.enable_auxiliary_loss:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()
        output_size = paddle.shape(fpn_feature_list[0])[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear',
                    align_corners=True))
        fusion_out = paddle.concat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.enable_auxiliary_loss:
            out.append(x)
            return out
        else:
            return [x]


class AlignedModule(nn.Layer):
    """
    The FAM module implementation.

    Args:
       inplane (int): Input channles of FAM module.
       outplane (int): Output channels of FAN module.
       kernel_size (int, optional): Kernel size of semantic flow convolution layer. Default: 3.
    """

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.down_l = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            outplane * 2,
            2,
            kernel_size=kernel_size,
            padding=1,
            bias_attr=False)

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

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        size = paddle.shape(low_feature)[2:]
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(paddle.concat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature


class AlignedModule_origin(nn.Layer):
    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()
        self.down_h = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.down_l = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            outplane * 2,
            2,
            kernel_size=kernel_size,
            padding=1,
            bias_attr=False)

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

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        low_feature_origin = low_feature
        size = paddle.shape(low_feature)[2:]
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(paddle.concat([h_feature, low_feature], 1))
        h_feature_up = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature_up + low_feature_origin


class AlignedModule_add(nn.Layer):
    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

    def forward(self, x):
        low_feature, h_feature = x
        size = paddle.shape(low_feature)[2:]
        up_feature = F.interpolate(h_feature, size=size, mode='bilinear')
        out = low_feature + up_feature
        return out


class AlignedModule_cat_conv(nn.Layer):
    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            2 * inplane, inplane, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        low_feature, h_feature = x
        size = paddle.shape(low_feature)[2:]
        h_feature = F.interpolate(h_feature, size=size, mode='bilinear')
        out = paddle.concat([low_feature, h_feature], axis=1)
        out = self.conv(out)
        return out


class AlignedModule_ch_atten_7_3(nn.Layer):
    '''combined ch atten, same as arm_7_3'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.conv_atten = nn.Conv2D(
            4 * inplane, inplane, kernel_size=1, bias_attr=None)
        self.bn_atten = layers.SyncBatchNorm(inplane)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy = paddle.concat([x, y], axis=1)  # n*2c*1*1
        xy_avg_pool = F.adaptive_avg_pool2d(xy, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy, 1)

        atten = paddle.concat([xy_avg_pool, xy_max_pool], axis=1)  # n*4c*1*1
        atten = self.conv_atten(atten)
        atten = F.sigmoid(self.bn_atten(atten))
        out = x * atten + y * (1 - atten)
        return out


class AlignedModule_sp_atten_1(nn.Layer):
    '''combined sp atten, same as arm_12_4'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()
        self.conv_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        mean_max_cat = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(mean_max_cat))  # n * 1 * h * w

        out = x * atten + y * (1 - atten)
        return out


class AlignedModule_sp_atten_12_6(nn.Layer):
    '''same as arm_12_6'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        inter_ch = inplane // 1
        self.sp_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * inplane,
                inter_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBNAct(
                inter_ch,
                inter_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBN(
                inter_ch, 1, kernel_size=1, padding=0, bias_attr=False))

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        atten = F.sigmoid(self.sp_atten(xy_cat))  # n * 1 * h * w
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_1(nn.Layer):
    '''ch atten + combined sp atten, same as arm_15_1 '''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.conv_x_atten = nn.Conv2D(
            inplane, inplane, kernel_size=1, bias_attr=False)
        self.conv_y_atten = nn.Conv2D(
            inplane, inplane, kernel_size=1, bias_attr=False)

        self.sp_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        x_atten = F.adaptive_avg_pool2d(x, 1)
        x_atten = F.sigmoid(self.conv_x_atten(x_atten))
        x = x * x_atten

        y_atten = F.adaptive_avg_pool2d(y, 1)
        y_atten = F.sigmoid(self.conv_y_atten(y_atten))
        y = y * y_atten

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max], axis=1)
        sp_atten = F.sigmoid(self.sp_atten(xy_mean_max_cat))  # n * 1 * h * w

        out = sp_atten * x + (1 - sp_atten) * y
        return out


class AlignedModule_ch_sp_atten_16_1(nn.Layer):
    '''combined ch atten + combined sp atten, same as arm_16_1, two sigmoid'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.ch_atten = nn.Conv2D(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = F.sigmoid(self.ch_atten(pool_cat))  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = F.sigmoid(self.sp_atten(xy_mean_max_cat))  # n * 1 * h * w

        atten = sp_atten * ch_atten
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_17_1(nn.Layer):
    '''combined ch atten + combined sp atten, same as arm_17_1, one sigmoid'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.ch_atten = nn.Conv2D(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        atten = F.sigmoid(sp_atten * ch_atten)
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_17_1_out_conv(nn.Layer):
    '''combined ch atten + combined sp atten, same as arm_17_1, one sigmoid'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.ch_atten = layers.ConvBN(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            inplane, inplane, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        atten = F.sigmoid(sp_atten * ch_atten)
        out = atten * x + (1 - atten) * y
        out = self.conv_out(out)
        return out


class AlignedModule_ch_sp_atten_17_1_atten_conv(nn.Layer):
    ''''''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.ch_atten = layers.ConvBN(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.conv_atten = layers.ConvBN(
            inplane, inplane, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        atten = F.sigmoid(self.conv_atten(sp_atten * ch_atten))
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_17_1_sfnet(AlignedModule_origin):
    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__(inplane, outplane, kernel_size)

        self.ch_atten = layers.ConvBN(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = layers.ConvBN(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        low_feature_origin = low_feature
        size = paddle.shape(low_feature)[2:]
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(paddle.concat([h_feature, low_feature], 1))
        h_feature_up = self.flow_warp(h_feature_orign, flow, size=size)

        x = low_feature_origin
        y = h_feature_up
        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = self.sp_atten(xy_mean_max_cat)  # n * 1 * h * w

        atten = F.sigmoid(sp_atten * ch_atten)
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_17_7_0(nn.Layer):
    '''combined ch atten + combined sp atten, same as arm_17_7, one sigmoid'''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        inter_ch = inplane // 1
        self.ch_atten = nn.Sequential(
            layers.ConvBNReLU(
                2 * inplane, inter_ch, kernel_size=1, bias_attr=False),
            nn.Conv2D(inter_ch, inplane, kernel_size=1, bias_attr=False),
        )

        inter_ch = inplane // 1
        self.sp_atten = nn.Sequential(
            layers.ConvBNReLU(
                2 * inplane,
                inter_ch,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            layers.ConvBNReLU(
                inter_ch, inter_ch, kernel_size=3, padding=1, bias_attr=False),
            nn.Conv2D(inter_ch, 1, kernel_size=1, padding=0, bias_attr=False))

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
        else:
            xy_avg_pool = paddle.mean(xy_cat, axis=[2, 3], keepdim=True)
        ch_atten = self.ch_atten(xy_avg_pool)  # n * c * 1 * 1

        sp_atten = self.sp_atten(xy_cat)  # n * 1 * h * w

        atten = F.sigmoid(sp_atten * ch_atten)
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_17_7_5(nn.Layer):
    ''''''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        inter_ch = inplane // 1
        self.ch_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * inplane,
                inter_ch,
                kernel_size=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBN(inter_ch, inplane, kernel_size=1, bias_attr=False))

        inter_ch = inplane // 1
        self.sp_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * inplane,
                inter_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBNAct(
                inter_ch,
                inter_ch,
                kernel_size=3,
                padding=1,
                act_type='leakyrelu',
                bias_attr=False),
            layers.ConvBN(
                inter_ch, 1, kernel_size=1, padding=0, bias_attr=False))

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        else:
            xy_avg_pool = paddle.mean(xy_cat, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy_cat, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = self.ch_atten(pool_cat)  # n * c * 1 * 1

        sp_atten = self.sp_atten(xy_cat)  # n * 1 * h * w

        atten = F.sigmoid(sp_atten * ch_atten)
        out = atten * x + (1 - atten) * y
        return out


class AlignedModule_ch_sp_atten_18_1(nn.Layer):
    ''''''

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()

        self.ch_atten = nn.Conv2D(
            4 * inplane, inplane, kernel_size=1, bias_attr=False)
        self.sp_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs):
        x, y = inputs

        size = paddle.shape(x)[2:]
        y = F.interpolate(y, size=size, mode='bilinear')

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w

        if self.training:
            xy_avg_pool = F.adaptive_avg_pool2d(xy_cat, 1)
            xy_max_pool = F.adaptive_max_pool2d(xy_cat, 1)
        else:
            xy_avg_pool = paddle.mean(xy_cat, axis=[2, 3], keepdim=True)
            xy_max_pool = paddle.max(xy_cat, axis=[2, 3], keepdim=True)
        pool_cat = paddle.concat([xy_avg_pool, xy_max_pool],
                                 axis=1)  # n * 4c * 1 * 1
        ch_atten = F.sigmoid(self.ch_atten(pool_cat))  # n * c * 1 * 1

        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        xy_mean_max_cat = paddle.concat([xy_mean, xy_max],
                                        axis=1)  # n * 2 * h * w
        sp_atten = F.sigmoid(self.sp_atten(xy_mean_max_cat))  # n * 1 * h * w

        # equal to out = (ch_atten + sp_atten) * x + (2 - ch_atten - sp_atten) * y
        out = ch_atten * x + (1 - ch_atten) * y + sp_atten * x + (
            1 - sp_atten) * y
        return out
