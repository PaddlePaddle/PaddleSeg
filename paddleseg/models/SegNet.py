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


@manager.MODELS.add_component
class SegNet(nn.Layer):
    """
    The SegNet implementation based on PaddlePaddle.

    The original article refers to
    Badrinarayanan, Vijay，Kendall, Alex，Cipolla, Roberto . "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
    (https://arxiv.org/pdf/1511.00561.pdf).

    Args:
        num_classes (int): The unique number of target classes.
    """
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        self.encoder = Encoder(input_channels=3)

        self.deco1 = nn.Sequential(
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1)
        )
        self.deco2 = nn.Sequential(
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,256,3,stride=1,paddling=1)
        )
        self.deco3 = nn.Sequential(
            layers.ConvBNReLU(256,256,3,stride=1,paddling=1),
            layers.ConvBNReLU(256,256,3,stride=1,paddling=1),
            layers.ConvBNReLU(256,128,3,stride=1,paddling=1)
        )
        self.deco4 = nn.Sequential(
            layers.ConvBNReLU(128,128,3,stride=1,paddling=1),
            layers.ConvBNReLU(128,128,3,stride=1,paddling=1),
            layers.ConvBNReLU(128,64,3,stride=1,paddling=1)
        )
        self.deco5 = nn.Sequential(
            layers.ConvBNReLU(64,64,3,stride=1,paddling=1),
            nn.Conv2D(64, num_classes, kernel_size=3, stride=1, padding=1),
        )
        self.init_weight()
    
    def init_weight(self):
      """Initialize the parameters of model parts."""
      for sublayer in self.sublayers():
        if isinstance(sublayer, nn.Conv2D):
          param_init.normal_init(sublayer.weight, std=0.001)
        elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
          param_init.constant_init(sublayer.weight, value=1.0)
          param_init.constant_init(sublayer.bias, value=0.0)

    def forward(self, x):
        x = self.encoder(x)
        y = []
        x = F.interpolate(x, paddle.to_tensor([[22],[30]],dtype='float32'), mode='bicubic')
        x = self.deco1(x)
        x = F.interpolate(x, paddle.to_tensor([[45],[60]],dtype='float32'), mode='bicubic')
        x = self.deco2(x)
        x = F.interpolate(x, paddle.to_tensor([[90],[120]],dtype='float32'), mode='bicubic')
        x = self.deco3(x)
        x = F.interpolate(x, paddle.to_tensor([[180],[240]],dtype='float32'), mode='bicubic')
        x = self.deco4(x)
        x = F.interpolate(x, paddle.to_tensor([[360],[480]],dtype='float32'), mode='bicubic')
        x = self.deco5(x)
        y.append(x)

        return y


class Encoder(nn.Layer):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.enco1 = nn.Sequential(
            layers.ConvBNReLU(input_channels,64,3,stride=1,paddling=1),
            layers.ConvBNReLU(64,64,3,stride=1,paddling=1)
        )
        self.enco2 = nn.Sequential(
            layers.ConvBNReLU(64,128,3,stride=1,paddling=1),
            layers.ConvBNReLU(128,128,3,stride=1,paddling=1)
        )
        self.enco3 = nn.Sequential(
            layers.ConvBNReLU(128,256,3,stride=1,paddling=1),
            layers.ConvBNReLU(256,256,3,stride=1,paddling=1),
            layers.ConvBNReLU(256,256,3,stride=1,paddling=1)
        )
        self.enco4 = nn.Sequential(
            layers.ConvBNReLU(256,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1)
        )
        self.enco5 = nn.Sequential(
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1),
            layers.ConvBNReLU(512,512,3,stride=1,paddling=1)
        )

    def forward(self, x):
        x = self.enco1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)  # 保留最大值的位置
        x = self.enco2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        x = self.enco3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        x = self.enco4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        x = self.enco5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)

        return x

