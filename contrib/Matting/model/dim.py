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

from collections import defaultdict
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg import utils
from paddleseg.cvlibs import manager

from .loss import MRSD


@manager.MODELS.add_component
class DIM(nn.Layer):
    """
    The DIM implementation based on PaddlePaddle.

    The original article refers to
    Ning Xu, et, al. "Deep Image Matting"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        backbone: backbone model.
        stage (int, optional): The stage of model. Defautl: 3.
        decoder_input_channels(int, optional): The channel of decoder input. Default: 512.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.

    """

    def __init__(self,
                 backbone,
                 stage=3,
                 decoder_input_channels=512,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.stage = stage

        decoder_output_channels = [64, 128, 256, 512]
        self.decoder = Decoder(
            input_channels=decoder_input_channels,
            output_channels=decoder_output_channels)
        if self.stage == 2:
            for param in self.backbone.parameters():
                param.stop_gradient = True
            for param in self.decoder.parameters():
                param.stop_gradient = True
        if self.stage >= 2:
            self.refine = Refine()
        self.init_weight()

    def forward(self, inputs):
        input_shape = paddle.shape(inputs['img'])[-2:]
        x = paddle.concat([inputs['img'], inputs['trimap'] / 255], axis=1)
        fea_list = self.backbone(x)

        # decoder stage
        up_shape = []
        for i in range(5):
            up_shape.append(paddle.shape(fea_list[i])[-2:])
        alpha_raw = self.decoder(fea_list, up_shape)
        alpha_raw = F.interpolate(
            alpha_raw, input_shape, mode='bilinear', align_corners=False)
        logit_dict = {'alpha_raw': alpha_raw}
        if self.stage < 2:
            return logit_dict

        if self.stage >= 2:
            # refine stage
            refine_input = paddle.concat([inputs['img'], alpha_raw], axis=1)
            alpha_refine = self.refine(refine_input)

            # finally alpha
            alpha_pred = alpha_refine + alpha_raw
            alpha_pred = F.interpolate(
                alpha_pred, input_shape, mode='bilinear', align_corners=False)
            if not self.training:
                alpha_pred = paddle.clip(alpha_pred, min=0, max=1)
            logit_dict['alpha_pred'] = alpha_pred
        if self.training:
            return logit_dict
        else:
            return alpha_pred

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            loss_func_dict = defaultdict(list)
            loss_func_dict['alpha_raw'].append(MRSD())
            loss_func_dict['comp'].append(MRSD())
            loss_func_dict['alpha_pred'].append(MRSD())

        loss = {}
        mask = label_dict['trimap'] == 128
        loss['all'] = 0

        if self.stage != 2:
            loss['alpha_raw'] = loss_func_dict['alpha_raw'][0](
                logit_dict['alpha_raw'], label_dict['alpha'], mask)
            loss['alpha_raw'] = 0.5 * loss['alpha_raw']
            loss['all'] = loss['all'] + loss['alpha_raw']

        if self.stage == 1 or self.stage == 3:
            comp_pred = logit_dict['alpha_raw'] * label_dict['fg'] + \
                (1 - logit_dict['alpha_raw']) * label_dict['bg']
            loss['comp'] = loss_func_dict['comp'][0](comp_pred,
                                                     label_dict['img'], mask)
            loss['comp'] = 0.5 * loss['comp']
            loss['all'] = loss['all'] + loss['comp']

        if self.stage == 2 or self.stage == 3:
            loss['alpha_pred'] = loss_func_dict['alpha_pred'][0](
                logit_dict['alpha_pred'], label_dict['alpha'], mask)
            loss['all'] = loss['all'] + loss['alpha_pred']

        return loss

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


# bilinear interpolate skip connect
class Up(nn.Layer):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            input_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias_attr=False)

    def forward(self, x, skip, output_shape):
        x = F.interpolate(
            x, size=output_shape, mode='bilinear', align_corners=False)
        x = x + skip
        x = self.conv(x)
        x = F.relu(x)

        return x


class Decoder(nn.Layer):
    def __init__(self, input_channels, output_channels=(64, 128, 256, 512)):
        super().__init__()
        self.deconv6 = nn.Conv2D(
            input_channels, input_channels, kernel_size=1, bias_attr=False)
        self.deconv5 = Up(input_channels, output_channels[-1])
        self.deconv4 = Up(output_channels[-1], output_channels[-2])
        self.deconv3 = Up(output_channels[-2], output_channels[-3])
        self.deconv2 = Up(output_channels[-3], output_channels[-4])
        self.deconv1 = Up(output_channels[-4], 64)

        self.alpha_conv = nn.Conv2D(
            64, 1, kernel_size=5, padding=2, bias_attr=False)

    def forward(self, fea_list, shape_list):
        x = fea_list[-1]
        x = self.deconv6(x)
        x = self.deconv5(x, fea_list[4], shape_list[4])
        x = self.deconv4(x, fea_list[3], shape_list[3])
        x = self.deconv3(x, fea_list[2], shape_list[2])
        x = self.deconv2(x, fea_list[1], shape_list[1])
        x = self.deconv1(x, fea_list[0], shape_list[0])
        alpha = self.alpha_conv(x)
        alpha = F.sigmoid(alpha)

        return alpha


class Refine(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(
            4, 64, kernel_size=3, padding=1, bias_attr=False)
        self.conv2 = layers.ConvBNReLU(
            64, 64, kernel_size=3, padding=1, bias_attr=False)
        self.conv3 = layers.ConvBNReLU(
            64, 64, kernel_size=3, padding=1, bias_attr=False)
        self.alpha_pred = layers.ConvBNReLU(
            64, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        alpha = self.alpha_pred(x)

        return alpha
