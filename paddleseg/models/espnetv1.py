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
import os

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)

 
@manager.MODELS.add_component
class ESPNetV1(nn.Layer):
    """
    The ESPNetV1 implementation based on PaddlePaddle.
    The original article refers to
    Sachin Mehta1, Mohammad Rastegari, Anat Caspi, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation"
    (https://arxiv.org/abs/1803.06815).
    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): Number of input channels. Default: 3.
        level2_depth (int, optional): Depth of DilatedResidualBlock. Default: 2.
        level3_depth (int, optional): Depth of DilatedResidualBlock. Default: 3.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self, num_classes, in_channels=3, level2_depth=2, level3_depth=3, pretrained=None):
        super().__init__()
        self.encoder = ESPNetEncoder(num_classes, in_channels, level2_depth, level3_depth)

        self.level3_up = nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False)
        self.br3 = SyncBatchNorm(num_classes)
        self.level2_proj = Conv(in_channels + 128, num_classes, 1, 1)
        self.combine_l2_l3 = nn.Sequential(
            BNPReLU(2 * num_classes),
            DilatedResidualBlock(2 * num_classes, num_classes, residual=False),
        )
        self.level2_up = nn.Sequential(
            nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False),
            BNPReLU(num_classes),
        ) 
        self.out_proj = ConvBNPReLU(16 + in_channels + num_classes, num_classes, 3, 1)
        self.out_up = nn.Conv2DTranspose(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias_attr=False)
    
    def forward(self, x):
        p1, p2, p3 = self.encoder(x) # shape [N, C, H/2, W/2]  [N, C, H/4, W/4]  [N, C, H/8, W/8] 
        up_p3 = self.level3_up(p3) # [N, C, H/4, W/4]

        combine = self.combine_l2_l3(paddle.concat([up_p3, p2], axis=1))
        up_p2 = self.level2_up(combine)  # [N, C, H/2, W/2]

        combine = self.out_proj(paddle.concat([up_p2, p1], axis=1))  # shape [N, C, H/2, W/2]
        out = self.out_up(combine)
        return [out]


class ConvBNPReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = SyncBatchNorm(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BNPReLU(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.bn = SyncBatchNorm(channels)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvDilated(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSampler(nn.Layer):
    """
    Down sampler.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 5
        remain_channels = out_channels - branch_channels * 4
        self.conv1 = Conv(in_channels, branch_channels, 3, 2)
        self.d_conv1 = ConvDilated(branch_channels, remain_channels, 3, 1, 1)
        self.d_conv2 = ConvDilated(branch_channels, branch_channels, 3, 1, 2)
        self.d_conv4 = ConvDilated(branch_channels, branch_channels, 3, 1, 4)
        self.d_conv8 = ConvDilated(branch_channels, branch_channels, 3, 1, 8)
        self.d_conv16 = ConvDilated(branch_channels, branch_channels, 3, 1, 16)
        self.bn = SyncBatchNorm(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        d1 = self.d_conv1(x)
        d2 = self.d_conv2(x)
        d4 = self.d_conv4(x)
        d8 = self.d_conv8(x)
        d16 = self.d_conv16(x)

        feat1 = d2
        feat2 = feat1 + d4
        feat3 = feat2 + d8
        feat4 = feat3 + d16

        feat = paddle.concat([d1, feat1, feat2, feat3, feat4], axis=1)
        out = self.bn(feat)
        out = self.act(out)
        return out
        

class DilatedResidualBlock(nn.Layer):
    '''
    ESP block, principle: reduce -> split -> transform -> merge
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        residual (bool, optional): Add a residual connection through identity operation. Default: True.
    '''
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        branch_channels = out_channels // 5
        remain_channels = out_channels - branch_channels * 4
        self.conv1 = Conv(in_channels, branch_channels, 1, 1)
        self.d_conv1 = ConvDilated(branch_channels, remain_channels, 3, 1, 1)
        self.d_conv2 = ConvDilated(branch_channels, branch_channels, 3, 1, 2)
        self.d_conv4 = ConvDilated(branch_channels, branch_channels, 3, 1, 4)
        self.d_conv8 = ConvDilated(branch_channels, branch_channels, 3, 1, 8)
        self.d_conv16 = ConvDilated(branch_channels, branch_channels, 3, 1, 16)
        self.bn = BNPReLU(out_channels)
        self.residual = residual

    def forward(self, x):
        x_proj = self.conv1(x)
        d1 = self.d_conv1(x_proj)
        d2 = self.d_conv2(x_proj)
        d4 = self.d_conv4(x_proj)
        d8 = self.d_conv8(x_proj)
        d16 = self.d_conv16(x_proj)

        feat1 = d2
        feat2 = feat1 + d4
        feat3 = feat2 + d8
        feat4 = feat3 + d16

        feat = paddle.concat([d1, feat1, feat2, feat3, feat4], axis=1)

        if self.residual:
            feat = feat + x
        out = self.bn(feat)
        return out


class ESPNetEncoder(nn.Layer):
    '''
    The ESPNet-C implementation based on PaddlePaddle.
    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): Number of input channels. Default: 3.
        level2_depth (int, optional): Depth of DilatedResidualBlock. Default: 5.
        level3_depth (int, optional): Depth of DilatedResidualBlock. Default: 3.
    '''
    def __init__(self, num_classes, in_channels=3, level2_depth=5, level3_depth=3):
        super().__init__()
        self.level1 = ConvBNPReLU(in_channels, 16, 3, 2)
        self.br1 = BNPReLU(in_channels + 16)
        self.proj1 = ConvBNPReLU(in_channels + 16, num_classes, 1)

        self.level2_0 = DownSampler(in_channels + 16, 64)
        self.level2 = nn.Sequential(
            *[DilatedResidualBlock(64, 64) for i in range(level2_depth)]
        )
        self.br2 = BNPReLU(in_channels + 128)
        self.proj2 = ConvBNPReLU(in_channels + 128, num_classes, 1)

        self.level3_0 = DownSampler(in_channels + 128, 128)
        self.level3 = nn.Sequential(
            *[DilatedResidualBlock(128, 128) for i in range(level3_depth)]
        )
        self.br3 = BNPReLU(256)
        self.proj3 = ConvBNPReLU(256, num_classes, 1)

    def forward(self, x):
        f1 = self.level1(x)
        down2 = F.adaptive_avg_pool2d(x, output_size=f1.shape[2:])
        feat1 = paddle.concat([f1, down2], axis=1)  # N, 19, H, W
        feat1 = self.br1(feat1)
        p1 = self.proj1(feat1)

        f2_res = self.level2_0(feat1)
        f2 = self.level2(f2_res)
        down4 = F.adaptive_avg_pool2d(x, output_size=f2.shape[2:])
        feat2 = paddle.concat([f2, f2_res, down4], axis=1)  # N, 3 + 128, H, W
        feat2 = self.br2(feat2)
        p2 = self.proj2(feat2)

        f3_res = self.level3_0(feat2)
        f3 = self.level3(f3_res)
        down8 = F.adaptive_avg_pool2d(x, output_size=f3.shape[2:])
        feat3 = paddle.concat([f3, f3_res], axis=1)   # N, 256, H, W
        feat3 = self.br3(feat3)
        p3 = self.proj3(feat3)

        return p1, p2, p3


if __name__ == '__main__':
    model = ESPNetV1(19, 3, 2, 8)
    paddle.summary(model, (4, 3, 256, 256))