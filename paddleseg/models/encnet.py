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
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class ENCNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 num_codes=32,
                 mid_channels=512,
                 up_scale=8,
                 in_index=[1, 2, 3],
                 use_se_loss=True,
                 add_lateral=False,
                 pretrained=None):
        super().__init__()
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.backbone = backbone
        self.in_index = in_index
        self.up_scale = up_scale
        in_channels = [self.backbone.feat_channels[index] for index in in_index]

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
        self.cls_seg = nn.Sequential(nn.Conv2D(mid_channels, num_classes, 1), )

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
        feats = self.backbone(inputs)
        fcn_feat = feats[2]

        temp_feats = []
        for i in self.in_index:
            temp_feats.append(feats[i])
        feats = temp_feats
        feat = self.bottleneck(feats[-1])

        if self.add_lateral:
            laterals = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                laterals.append(
                    F.interpolate(lateral_conv(feats[i]),
                                  size=paddle.shape(feat)[2:],
                                  mode='bilinear',
                                  align_corners=False))
            feat = self.fusion(paddle.concat([feat, *laterals], 1))
        encode_feat, feat = self.enc_module(feat)
        out = self.cls_seg(feat)
        out = F.interpolate(out,
                            scale_factor=self.up_scale,
                            mode='bilinear',
                            align_corners=False)
        output = [out]
        if self.training:
            fcn_out = self.fcn_head(fcn_feat)
            fcn_out = F.interpolate(fcn_out,
                                    scale_factor=self.up_scale,
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
        batch_size = paddle.shape(x)
        reshaped_scale = scale.reshape([1, 1, num_codes])
        # expanded_x = paddle.expand(x.unsqueeze(2), [batch_size, paddle.shape(x)[1], num_codes, channels])
        expanded_x = paddle.tile(x.unsqueeze(2), [1, 1, num_codes, 1])
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])

        scaled_l2_norm = paddle.multiply(
            reshaped_scale,
            (expanded_x - reshaped_codewords).pow(2).sum(axis=3))
        return scaled_l2_norm

    def aggregate(self, assignment_weights, x, codewords):
        num_codes, channels = paddle.shape(codewords)
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])
        batch_size = paddle.shape(x)[0]
        # expanded_x = paddle.expand(x.unsqueeze(2), [batch_size, paddle.shape(x)[1], num_codes, channels])
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
        self.num_codes = num_codes

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection)

        encoding_feat = encoding_feat.mean(axis=1)
        batch_size, channels, _, _ = paddle.shape(x)

        gamma = self.fc(encoding_feat)
        y = gamma.reshape([batch_size, self.in_channels, 1, 1])
        output = F.relu(x + x * y)
        return encoding_feat, output
