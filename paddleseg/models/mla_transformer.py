# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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


class MLAHeads(nn.Layer):
    def __init__(self, mlahead_channels=128):
        super(MLAHeads, self).__init__()
        self.head2 = nn.Sequential(
            layers.ConvBNReLU(
                mlahead_channels * 2,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            layers.ConvBNReLU(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False))
        self.head3 = nn.Sequential(
            layers.ConvBNReLU(
                mlahead_channels * 2,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            layers.ConvBNReLU(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False))
        self.head4 = nn.Sequential(
            layers.ConvBNReLU(
                mlahead_channels * 2,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            layers.ConvBNReLU(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False))
        self.head5 = nn.Sequential(
            layers.ConvBNReLU(
                mlahead_channels * 2,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            layers.ConvBNReLU(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False))

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(
            self.head2(mla_p2),
            size=(4 * mla_p2.shape[3], 4 * mla_p2.shape[3]),
            mode='bilinear',
            align_corners=True)
        head3 = F.interpolate(
            self.head3(mla_p3),
            size=(4 * mla_p3.shape[3], 4 * mla_p3.shape[3]),
            mode='bilinear',
            align_corners=True)
        head4 = F.interpolate(
            self.head4(mla_p4),
            size=(4 * mla_p4.shape[3], 4 * mla_p4.shape[3]),
            mode='bilinear',
            align_corners=True)
        head5 = F.interpolate(
            self.head5(mla_p5),
            size=(4 * mla_p5.shape[3], 4 * mla_p5.shape[3]),
            mode='bilinear',
            align_corners=True)

        return paddle.concat([head2, head3, head4, head5], axis=1)


@manager.MODELS.add_component
class MLATransformer(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 backbone,
                 mlahead_channels=128,
                 aux_channels=256,
                 norm_layer=nn.BatchNorm2D,
                 pretrained=None,
                 **kwargs):
        super(MLATransformer, self).__init__()

        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone = backbone

        self.mlahead = MLAHeads(mlahead_channels=self.mlahead_channels)
        self.cls = nn.Conv2D(
            4 * self.mlahead_channels, self.num_classes, 3, padding=1)

        self.conv0 = layers.ConvBNReLU(
            self.in_channels[0],
            self.in_channels[0] * 2,
            3,
            padding=1,
            bias_attr=False)
        self.conv1 = layers.ConvBNReLU(
            self.in_channels[1],
            self.in_channels[1],
            3,
            padding=1,
            bias_attr=False)
        self.conv21 = layers.ConvBNReLU(
            self.in_channels[2],
            self.in_channels[2],
            3,
            padding=1,
            bias_attr=False)
        self.conv22 = layers.ConvBNReLU(
            self.in_channels[2],
            self.in_channels[2] // 2,
            3,
            padding=1,
            bias_attr=False)
        self.conv31 = layers.ConvBNReLU(
            self.in_channels[3],
            self.in_channels[3],
            3,
            padding=1,
            bias_attr=False)
        self.conv32 = layers.ConvBNReLU(
            self.in_channels[3],
            self.in_channels[3] // 2,
            3,
            padding=1,
            bias_attr=False)
        self.conv33 = layers.ConvBNReLU(
            self.in_channels[3] // 2,
            self.in_channels[3] // 4,
            3,
            padding=1,
            bias_attr=False)

        self.aux_head = nn.Sequential(
            layers.ConvBN(
                in_channels=self.in_channels[2],
                out_channels=aux_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.Conv2D(
                in_channels=aux_channels,
                out_channels=self.num_classes,
                kernel_size=1, ))

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        inputs = self.backbone(x)

        inputs0 = self.conv0(inputs[0])
        inputs1 = F.interpolate(
            self.conv1(inputs[1]),
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=True)
        inputs2 = F.interpolate(
            self.conv21(inputs[2]),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        inputs2 = F.interpolate(
            self.conv22(inputs2),
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=True)
        inputs3 = F.interpolate(
            self.conv31(inputs[3]),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        inputs3 = F.interpolate(
            self.conv32(inputs3),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        inputs3 = F.interpolate(
            self.conv33(inputs3),
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=True)
        inputs2 = inputs2 + inputs3
        inputs1 = inputs1 + inputs2
        inputs0 = inputs0 + inputs1

        feats = self.mlahead(inputs0, inputs1, inputs2, inputs3)
        logit = self.cls(feats)
        logit_list = [logit]

        if self.training:
            logit_list.append(self.aux_head(inputs[2]))

        logit_list = [
            F.interpolate(
                logit, paddle.shape(x)[2:], mode='bilinear', align_corners=True)
            for logit in logit_list
        ]
        return logit_list
