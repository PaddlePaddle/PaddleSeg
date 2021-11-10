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

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.backbones import resnet_vd
from paddleseg.models import deeplab
from paddleseg.utils import utils


class GSCNNHead(nn.Layer):
    """
    The GSCNNHead implementation based on PaddlePaddle.
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the last one will be taken as input of ASPP component; the second to fourth
            will be taken as input for GCL component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 1, 2, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, feature map of the fourth
            stage as input of ASPP, and the feature map of the second to fourth stage as input of GCL.
        backbone_channels (tuple): The channels of output of backbone.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, align_corners):
        super().__init__()
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners

        self.dsn1 = nn.Conv2D(
            backbone_channels[backbone_indices[1]], 1, kernel_size=1)
        self.dsn2 = nn.Conv2D(
            backbone_channels[backbone_indices[2]], 1, kernel_size=1)
        self.dsn3 = nn.Conv2D(
            backbone_channels[backbone_indices[3]], 1, kernel_size=1)

        self.res1 = resnet_vd.BasicBlock(64, 64, stride=1)
        self.d1 = nn.Conv2D(64, 32, kernel_size=1)
        self.gate1 = GatedSpatailConv2d(32, 32)
        self.res2 = resnet_vd.BasicBlock(32, 32, stride=1)
        self.d2 = nn.Conv2D(32, 16, kernel_size=1)
        self.gate2 = GatedSpatailConv2d(16, 16)
        self.res3 = resnet_vd.BasicBlock(16, 16, stride=1)
        self.d3 = nn.Conv2D(16, 8, kernel_size=1)
        self.gate3 = GatedSpatailConv2d(8, 8)
        self.fuse = nn.Conv2D(8, 1, kernel_size=1, bias_attr=False)

        self.cw = nn.Conv2D(2, 1, kernel_size=1, bias_attr=False)

        self.aspp = ASPPModule(
            aspp_ratios=aspp_ratios,
            in_channels=backbone_channels[-1],
            out_channels=aspp_out_channels,
            align_corners=self.align_corners,
            image_pooling=True)

        self.decoder = deeplab.Decoder(
            num_classes=num_classes,
            in_channels=backbone_channels[0],
            align_corners=self.align_corners)

    def forward(self, x, feat_list, s_input):
        input_shape = paddle.shape(x)
        m1f = F.interpolate(
            s_input,
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        l1, l2, l3 = [
            feat_list[self.backbone_indices[i]]
            for i in range(1, len(self.backbone_indices))
        ]
        s1 = F.interpolate(
            self.dsn1(l1),
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        s2 = F.interpolate(
            self.dsn2(l2),
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        s3 = F.interpolate(
            self.dsn3(l3),
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # Get image gradient
        im_arr = x.numpy().transpose((0, 2, 3, 1))
        im_arr = ((im_arr * 0.5 + 0.5) * 255).astype(np.uint8)
        canny = np.zeros((input_shape[0], 1, input_shape[2], input_shape[3]))
        for i in range(input_shape[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = canny / 255
        canny = paddle.to_tensor(canny).astype('float32')
        canny.stop_gradient = True

        cs = self.res1(m1f)
        cs = F.interpolate(
            cs,
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d1(cs)
        cs = self.gate1(cs, s1)

        cs = self.res2(cs)
        cs = F.interpolate(
            cs,
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d2(cs)
        cs = self.gate2(cs, s2)

        cs = self.res3(cs)
        cs = F.interpolate(
            cs,
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = self.d3(cs)
        cs = self.gate3(cs, s3)

        cs = self.fuse(cs)
        cs = F.interpolate(
            cs,
            input_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        cs = F.sigmoid(cs)  # Ouput of shape stream

        return [
            cs,
        ]


class GatedSpatailConv2d(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False):
        super().__init__()
        self._gate_conv = nn.Sequential(
            layers.SyncBatchNorm(in_channels + 1),
            nn.Conv2D(in_channels + 1, in_channels + 1, kernel_size=1),
            nn.ReLU(), nn.Conv2D(in_channels + 1, 1, kernel_size=1),
            layers.SyncBatchNorm(1), nn.Sigmoid())
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)

    def forward(self, input_features, gating_features):
        cat = paddle.concat([input_features, gating_features], axis=1)
        alphas = self._gate_conv(cat)
        x = input_features * (alphas + 1)
        x = self.conv(x)
        return x


class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.
    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_sep_conv=False,
                 image_pooling=False):
        super().__init__()

        self.align_corners = align_corners
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = layers.SeparableConvBNReLU
            else:
                conv_func = layers.ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(output_size=(1, 1)),
                layers.ConvBNReLU(
                    in_channels, out_channels, kernel_size=1, bias_attr=False))
            out_size += 1
        self.image_pooling = image_pooling

        self.edge_conv = layers.ConvBNReLU(
            1, out_channels, kernel_size=1, bias_attr=False)
        out_size += 1

        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge):
        outputs = []
        x_shape = paddle.shape(x)
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(
                y,
                x_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                x_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(img_avg)

        edge_features = F.interpolate(
            edge,
            size=x_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        edge_features = self.edge_conv(edge_features)
        outputs.append(edge_features)

        x = paddle.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        return x
