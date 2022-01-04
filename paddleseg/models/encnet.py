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

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class ENCNet(nn.Layer):
    """
    The ENCNet implementation based on PaddlePaddle.

    The original article refers to
    Hang Zhang, Kristin Dana, et, al. "Context Encoding for Semantic Segmentation".

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        num_codes (int): The number of encoded words. Default: 32.
        mid_channels (int): The channels of middle layers. Default: 512.
        use_se_loss (int): Whether use semantic encoding loss. Default: True.
        add_lateral (int): Whether use lateral convolution layers. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[1, 2, 3],
                 num_codes=32,
                 mid_channels=512,
                 use_se_loss=True,
                 add_lateral=False,
                 pretrained=None):
        super().__init__()
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [
            self.backbone.feat_channels[index] for index in backbone_indices
        ]

        self.bottleneck = layers.ConvBNReLU(
            in_channels[-1],
            mid_channels,
            3,
            padding=1,
        )
        if self.add_lateral:
            self.lateral_convs = nn.LayerList()
            for in_ch in in_channels[:-1]:
                self.lateral_convs.append(
                    layers.ConvBNReLU(
                        in_ch,
                        mid_channels,
                        1,
                    ))
            self.fusion = layers.ConvBNReLU(
                len(in_channels) * mid_channels,
                mid_channels,
                3,
                padding=1,
            )

        self.enc_module = EncModule(mid_channels, num_codes)
        self.head = nn.Conv2D(mid_channels, num_classes, 1)

        self.fcn_head = layers.AuxLayer(self.backbone.feat_channels[2],
                                        mid_channels, num_classes)

        self.use_se_loss = use_se_loss
        if use_se_loss:
            self.se_layer = nn.Linear(mid_channels, num_classes)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        N, C, H, W = paddle.shape(inputs)
        feats = self.backbone(inputs)
        fcn_feat = feats[2]

        feats = [feats[i] for i in self.backbone_indices]
        feat = self.bottleneck(feats[-1])

        if self.add_lateral:
            laterals = []
            for j, lateral_conv in enumerate(self.lateral_convs):
                laterals.append(
                    F.interpolate(lateral_conv(feats[j]),
                                  size=paddle.shape(feat)[2:],
                                  mode='bilinear',
                                  align_corners=False))
            feat = self.fusion(paddle.concat([feat, *laterals], 1))
        encode_feat, feat = self.enc_module(feat)
        out = self.head(feat)
        out = F.interpolate(out,
                            size=[H, W],
                            mode='bilinear',
                            align_corners=False)
        output = [out]
        if self.training:
            fcn_out = self.fcn_head(fcn_feat)
            fcn_out = F.interpolate(fcn_out,
                                    size=[H, W],
                                    mode='bilinear',
                                    align_corners=False)
            output.append(fcn_out)
            if self.use_se_loss:
                se_out = self.se_layer(encode_feat)
                output.append(se_out)
            return output
        return output


class Encoding(nn.Layer):
    def __init__(self, channels, num_codes):
        super().__init__()
        self.channels, self.num_codes = channels, num_codes

        std = 1 / ((channels * num_codes)**0.5)
        self.codewords = self.create_parameter(
            shape=(num_codes, channels),
            default_initializer=nn.initializer.Uniform(-std, std),
        )
        self.scale = self.create_parameter(
            shape=(num_codes, ),
            default_initializer=nn.initializer.Uniform(-1, 0),
        )
        self.channels = channels

    def scaled_l2(self, x, codewords, scale):
        num_codes, channels = paddle.shape(codewords)
        reshaped_scale = scale.reshape([1, 1, num_codes])
        expanded_x = paddle.tile(x.unsqueeze(2), [1, 1, num_codes, 1])
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])

        scaled_l2_norm = paddle.multiply(
            reshaped_scale,
            (expanded_x - reshaped_codewords).pow(2).sum(axis=3))
        return scaled_l2_norm

    def aggregate(self, assignment_weights, x, codewords):
        num_codes, channels = paddle.shape(codewords)
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])
        expanded_x = paddle.tile(x.unsqueeze(2), [1, 1, num_codes, 1])

        encoded_feat = paddle.multiply(
            assignment_weights.unsqueeze(3),
            (expanded_x - reshaped_codewords)).sum(axis=1)
        encoded_feat = paddle.reshape(encoded_feat,
                                      [-1, self.num_codes, self.channels])
        return encoded_feat

    def forward(self, x):
        x_dims = x.ndim
        assert x_dims == 4, "The dimension of input tensor must equal 4, but got {}.".format(
            x_dims)
        assert paddle.shape(
            x
        )[1] == self.channels, "Encoding channels error, excepted {} but got {}.".format(
            self.channels,
            paddle.shape(x)[1])
        batch_size = paddle.shape(x)[0]
        x = x.reshape([batch_size, self.channels, -1]).transpose([0, 2, 1])
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords,
                                                      self.scale),
                                       axis=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


class EncModule(nn.Layer):
    def __init__(self, in_channels, num_codes):
        super().__init__()
        self.encoding_project = layers.ConvBNReLU(
            in_channels,
            in_channels,
            1,
        )
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1D(num_codes),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )
        self.in_channels = in_channels

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection)

        encoding_feat = encoding_feat.mean(axis=1)
        batch_size, _, _, _ = paddle.shape(x)

        gamma = self.fc(encoding_feat)
        y = gamma.reshape([batch_size, self.in_channels, 1, 1])
        output = F.relu(x + x * y)
        return encoding_feat, output
