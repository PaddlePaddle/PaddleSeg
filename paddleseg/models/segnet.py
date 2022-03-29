# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class SegNet(nn.Layer):
    """
    The SegNet implementation based on PaddlePaddle.
    The original article refers to
    Badrinarayanan, Vijay, et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
    (https://arxiv.org/pdf/1511.00561.pdf).
    Args:
        num_classes (int): The unique number of target classes.
    """

    def __init__(self, num_classes, pretrained=None):
        super().__init__()

        # Encoder Module

        self.enco1 = nn.Sequential(
            layers.ConvBNReLU(
                3, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1))

        self.enco2 = nn.Sequential(
            layers.ConvBNReLU(
                64, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1))

        self.enco3 = nn.Sequential(
            layers.ConvBNReLU(
                128, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 256, 3, padding=1))

        self.enco4 = nn.Sequential(
            layers.ConvBNReLU(
                256, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1))

        self.enco5 = nn.Sequential(
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1))

        # Decoder Module

        self.deco1 = nn.Sequential(
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1))

        self.deco2 = nn.Sequential(
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 256, 3, padding=1))

        self.deco3 = nn.Sequential(
            layers.ConvBNReLU(
                256, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 128, 3, padding=1))

        self.deco4 = nn.Sequential(
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 64, 3, padding=1))

        self.deco5 = nn.Sequential(
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            nn.Conv2D(
                64, num_classes, kernel_size=3, padding=1), )

        self.pretrained = pretrained

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        logit_list = []

        x = self.enco1(x)
        x, ind1 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size1 = x.shape

        x = self.enco2(x)
        x, ind2 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size2 = x.shape

        x = self.enco3(x)
        x, ind3 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size3 = x.shape

        x = self.enco4(x)
        x, ind4 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size4 = x.shape

        x = self.enco5(x)
        x, ind5 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size5 = x.shape

        x = F.max_unpool2d(
            x, indices=ind5, kernel_size=2, stride=2, output_size=size4)
        x = self.deco1(x)

        x = F.max_unpool2d(
            x, indices=ind4, kernel_size=2, stride=2, output_size=size3)
        x = self.deco2(x)

        x = F.max_unpool2d(
            x, indices=ind3, kernel_size=2, stride=2, output_size=size2)
        x = self.deco3(x)

        x = F.max_unpool2d(
            x, indices=ind2, kernel_size=2, stride=2, output_size=size1)
        x = self.deco4(x)

        x = F.max_unpool2d(x, indices=ind1, kernel_size=2, stride=2)
        x = self.deco5(x)

        logit_list.append(x)

        return logit_list
