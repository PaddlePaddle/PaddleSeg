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
from paddleseg import utils
from paddleseg.cvlibs import manager


@manager.MODELS.add_component
class DIM(nn.Layer):
    """
    The DIM implementation based on PaddlePaddle.

    The original article refers to
    Ning Xu, et, al. "Deep Image Matting"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:


    """

    def __init__(self, backbone, backbone_indices=(-1, ), pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.pretrained = pretrained

        self.decoder = Decoder(input_channels=512)
        self.refine = Refine()
        self.init_weight()

    def forward(self, inputs):
        x = paddle.concat([inputs['img'], inputs['trimap'].unsqueeze(1)],
                          axis=1)
        fea_list, ids_list = self.backbone(x)

        # decoder stage
        up_shape = []
        up_shape.append(x.shape[-2:])
        for i in range(4):
            up_shape.append(fea_list[i].shape[-2:])
        alpha_raw = self.decoder(fea_list[self.backbone_indices[0]], up_shape)

        # refine stage
        alpha_raw_ = alpha_raw * 255
        refine_input = paddle.concat([x[:, :3, :, :], alpha_raw_], axis=1)
        alpha_refine = self.refine(refine_input)

        # finally alpha
        alpha_pred = alpha_refine + alpha_raw
        alpha_pred = paddle.clip(alpha_pred, min=0, max=1)

        logit_dict = {'alpha_pred': alpha_pred, 'alpha_raw': alpha_raw}
        return logit_dict

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Up(nn.Layer):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            input_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias_attr=False)

    def forward(self, x, output_shape):
        x = F.interpolate(
            x, size=output_shape, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class Decoder(nn.Layer):
    def __init__(self, input_channels):
        super().__init__()
        self.deconv6 = nn.Conv2D(
            input_channels, 512, kernel_size=1, bias_attr=False)
        self.deconv5 = Up(512, 512)
        self.deconv4 = Up(512, 256)
        self.deconv3 = Up(256, 128)
        self.deconv2 = Up(128, 64)
        self.deconv1 = Up(64, 64)

        self.alpha_conv = nn.Conv2D(
            64, 1, kernel_size=5, padding=2, bias_attr=False)

    def forward(self, x, shape_list):
        x = self.deconv6(x)
        x = self.deconv5(x, shape_list[4])
        x = self.deconv4(x, shape_list[3])
        x = self.deconv3(x, shape_list[2])
        x = self.deconv2(x, shape_list[1])
        x = self.deconv1(x, shape_list[0])
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
        alpha = F.sigmoid(alpha)

        return alpha


if __name__ == "__main__":
    from vgg import VGG16
    backbone = VGG16(input_channels=4)
    model = DIM(backbone=backbone)

    model_input = paddle.randint(0, 256, (1, 4, 320, 320)).astype('float32')
    alpha_pred, alpha_raw = model(model_input)

    print(model)

    print(alpha_pred.shape, alpha_raw.shape)
