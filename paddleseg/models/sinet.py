# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Refer to the origin implementation: https://github.com/clovaai/c3_sinet/blob/master/models/SINet.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

CFG = [[[3, 1], [5, 1]], [[3, 1], [3, 1]], [[3, 1], [5, 1]], [[3, 1], [3, 1]],
       [[5, 1], [3, 2]], [[5, 2], [3, 4]], [[3, 1], [3, 1]], [[5, 1], [5, 1]],
       [[3, 2], [3, 4]], [[3, 1], [5, 2]]]


@manager.MODELS.add_component
class SINet(nn.Layer):
    """
    The SINet implementation based on PaddlePaddle.

    The original article refers to
    Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Nicolas Monet, Jihwan Bang, Nojun Kwak
    "SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules
    and Information Blocking Decoder", (https://arxiv.org/abs/1911.09099).

    Args:
        num_classes (int): The unique number of target classes.
        config (List, optional): The config for SINet. Defualt use the CFG.
        stage2_blocks (int, optional): The num of blocks in stage2. Default: 2.
        stage3_blocks (int, optional): The num of blocks in stage3. Default: 8.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes=2,
                 config=CFG,
                 stage2_blocks=2,
                 stage3_blocks=8,
                 pretrained=None):
        super().__init__()
        dim1 = 16
        dim2 = 48
        dim3 = 96

        self.encoder = SINetEncoder(config, num_classes, stage2_blocks,
                                    stage3_blocks)

        self.up = nn.UpsamplingBilinear2D(scale_factor=2)
        self.bn_3 = nn.BatchNorm(num_classes)

        self.level2_C = CBR(dim2, num_classes, 1, 1)
        self.bn_2 = nn.BatchNorm(num_classes)

        self.classifier = nn.Sequential(
            nn.UpsamplingBilinear2D(scale_factor=2),
            nn.Conv2D(
                num_classes, num_classes, 3, 1, 1, bias_attr=False))

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, input):
        output1 = self.encoder.level1(input)  # x2

        output2_0 = self.encoder.level2_0(output1)  # x4
        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.encoder.BR2(paddle.concat([output2_0, output2], 1))

        output3_0 = self.encoder.level3_0(output2_cat)  # x8
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)
        output3_cat = self.encoder.BR3(paddle.concat([output3_0, output3], 1))
        enc_final = self.encoder.classifier(output3_cat)  # x8

        dec_stage1 = self.bn_3(self.up(enc_final))  # x4
        stage1_confidence = paddle.max(F.softmax(dec_stage1), axis=1)
        stage1_gate = (1 - stage1_confidence).unsqueeze(1)

        dec_stage2_0 = self.level2_C(output2)  # x4
        dec_stage2 = self.bn_2(
            self.up(dec_stage2_0 * stage1_gate + dec_stage1))  # x2

        out = self.classifier(dec_stage2)  # x

        return [out]


def channel_shuffle(x, groups):
    x_shape = paddle.shape(x)
    batch_size, height, width = x_shape[0], x_shape[2], x_shape[3]
    num_channels = x.shape[1]
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = paddle.reshape(x=x, shape=[batch_size, num_channels, height, width])

    return x


class CBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2D(
            nIn,
            nOut, (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias_attr=False)
        self.bn = nn.BatchNorm(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class SeparableCBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2D(
                nIn,
                nIn, (kSize, kSize),
                stride=stride,
                padding=(padding, padding),
                groups=nIn,
                bias_attr=False),
            nn.Conv2D(
                nIn, nOut, kernel_size=1, stride=1, bias_attr=False), )
        self.bn = nn.BatchNorm(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class SqueezeBlock(nn.Layer):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()

        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size), )
        else:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size), nn.PReLU(exp_size))

    def forward(self, x):
        alpha = F.adaptive_avg_pool2d(x, [1, 1])
        alpha = paddle.squeeze(alpha, axis=[2, 3])
        alpha = self.dense(alpha)
        alpha = paddle.unsqueeze(alpha, axis=[2, 3])
        out = x * alpha
        return out


class SESeparableCBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, divide=2.0):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2D(
                nIn,
                nIn, (kSize, kSize),
                stride=stride,
                padding=(padding, padding),
                groups=nIn,
                bias_attr=False),
            SqueezeBlock(
                nIn, divide=divide),
            nn.Conv2D(
                nIn, nOut, kernel_size=1, stride=1, bias_attr=False), )

        self.bn = nn.BatchNorm(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Layer):
    '''
    This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Layer):
    '''
    This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2D(
            nIn,
            nOut, (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias_attr=False)
        self.bn = nn.BatchNorm(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Layer):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, group=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2D(
            nIn,
            nOut, (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias_attr=False,
            groups=group)

    def forward(self, input):
        output = self.conv(input)
        return output


class S2block(nn.Layer):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, avgsize):
        super().__init__()

        self.resolution_down = False
        if avgsize > 1:
            self.resolution_down = True
            self.down_res = nn.AvgPool2D(avgsize, avgsize)
            self.up_res = nn.UpsamplingBilinear2D(scale_factor=avgsize)
            self.avgsize = avgsize

        padding = int((kSize - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2D(
                nIn,
                nIn,
                kernel_size=(kSize, kSize),
                stride=1,
                padding=(padding, padding),
                groups=nIn,
                bias_attr=False),
            nn.BatchNorm(nIn))

        self.act_conv1x1 = nn.Sequential(
            nn.PReLU(nIn),
            nn.Conv2D(
                nIn, nOut, kernel_size=1, stride=1, bias_attr=False), )

        self.bn = nn.BatchNorm(nOut)

    def forward(self, input):
        if self.resolution_down:
            input = self.down_res(input)
        output = self.conv(input)

        output = self.act_conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.bn(output)


class S2module(nn.Layer):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, config=[[3, 1], [5, 1]]):
        super().__init__()

        group_n = len(config)
        assert group_n == 2
        n = int(nOut / group_n)
        n1 = nOut - group_n * n

        self.c1 = C(nIn, n, 1, 1, group=group_n)
        # self.c1 = C(nIn, n, 1, 1)

        for i in range(group_n):
            if i == 0:
                self.layer_0 = S2block(
                    n, n + n1, kSize=config[i][0], avgsize=config[i][1])
            else:
                self.layer_1 = S2block(
                    n, n, kSize=config[i][0], avgsize=config[i][1])

        self.BR = BR(nOut)
        self.add = add
        self.group_n = group_n

    def forward(self, input):
        output1 = self.c1(input)
        output1 = channel_shuffle(output1, self.group_n)
        res_0 = self.layer_0(output1)
        res_1 = self.layer_1(output1)
        combine = paddle.concat([res_0, res_1], 1)

        if self.add:
            combine = input + combine
        output = self.BR(combine)
        return output


class SINetEncoder(nn.Layer):
    def __init__(self, config, num_classes=2, stage2_blocks=2, stage3_blocks=8):
        super().__init__()
        assert stage2_blocks == 2
        dim1 = 16
        dim2 = 48
        dim3 = 96

        self.level1 = CBR(3, 12, 3, 2)

        self.level2_0 = SESeparableCBR(12, dim1, 3, 2, divide=1)

        self.level2 = nn.LayerList()
        for i in range(0, stage2_blocks):
            if i == 0:
                self.level2.append(
                    S2module(
                        dim1, dim2, config=config[i], add=False))
            else:
                self.level2.append(S2module(dim2, dim2, config=config[i]))
        self.BR2 = BR(dim2 + dim1)

        self.level3_0 = SESeparableCBR(dim2 + dim1, dim2, 3, 2, divide=2)
        self.level3 = nn.LayerList()
        for i in range(0, stage3_blocks):
            if i == 0:
                self.level3.append(
                    S2module(
                        dim2, dim3, config=config[2 + i], add=False))
            else:
                self.level3.append(S2module(dim3, dim3, config=config[2 + i]))
        self.BR3 = BR(dim3 + dim2)

        self.classifier = C(dim3 + dim2, num_classes, 1, 1)

    def forward(self, input):
        output1 = self.level1(input)  # x2

        output2_0 = self.level2_0(output1)  # x4
        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output3_0 = self.level3_0(
            self.BR2(paddle.concat([output2_0, output2], 1)))  # x8
        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.BR3(paddle.concat([output3_0, output3], 1))
        classifier = self.classifier(output3_cat)
        return classifier
