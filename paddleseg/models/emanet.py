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
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class EMANet(nn.Layer):
    """
    Expectation Maximization Attention Networks for Semantic Segmentation based on PaddlePaddle.

    The original article refers to
    Xia Li, et al. "Expectation-Maximization Attention Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1907.13426)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of output of backbone.
        ema_channels (int): EMA module channels.
        gc_channels (int): The input channels to Global Context Block.
        num_bases (int): Number of bases.
        stage_num (int): The iteration number for EM.
        momentum (float): The parameter for updating bases.
        concat_input (bool): Whether concat the input and output of convs before classification layer. Default: True
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 ema_channels=512,
                 gc_channels=256,
                 num_bases=64,
                 stage_num=3,
                 momentum=0.1,
                 concat_input=True,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.head = EMAHead(num_classes, in_channels, ema_channels, gc_channels,
                            num_bases, stage_num, momentum, concat_input,
                            enable_auxiliary_loss)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class EMAHead(nn.Layer):
    """
    The EMANet head.

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (tuple): The number of input channels.
        ema_channels (int): EMA module channels.
        gc_channels (int): The input channels to Global Context Block.
        num_bases (int): Number of bases.
        stage_num (int): The iteration number for EM.
        momentum (float): The parameter for updating bases.
        concat_input (bool): Whether concat the input and output of convs before classification layer. Default: True
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 ema_channels,
                 gc_channels,
                 num_bases,
                 stage_num,
                 momentum,
                 concat_input=True,
                 enable_auxiliary_loss=True):
        super(EMAHead, self).__init__()

        self.in_channels = in_channels[-1]
        self.concat_input = concat_input
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.emau = EMAU(ema_channels, num_bases, stage_num, momentum=momentum)
        self.ema_in_conv = layers.ConvBNReLU(
            in_channels=self.in_channels,
            out_channels=ema_channels,
            kernel_size=3)
        self.ema_mid_conv = nn.Conv2D(ema_channels, ema_channels, kernel_size=1)
        self.ema_out_conv = layers.ConvBNReLU(
            in_channels=ema_channels, out_channels=ema_channels, kernel_size=1)
        self.bottleneck = layers.ConvBNReLU(
            in_channels=ema_channels, out_channels=gc_channels, kernel_size=3)
        self.cls = nn.Sequential(
            nn.Dropout2D(p=0.1), nn.Conv2D(gc_channels, num_classes, 1))
        self.aux = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=1024, out_channels=256, kernel_size=3),
            nn.Dropout2D(p=0.1), nn.Conv2D(256, num_classes, 1))
        if self.concat_input:
            self.conv_cat = layers.ConvBNReLU(
                self.in_channels + gc_channels, gc_channels, kernel_size=3)

    def forward(self, feat_list):
        C3, C4 = feat_list
        feats = self.ema_in_conv(C4)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.emau(feats)
        recon = F.relu(recon)
        recon = self.ema_out_conv(recon)
        output = F.relu(identity + recon)
        output = self.bottleneck(output)
        if self.concat_input:
            output = self.conv_cat(paddle.concat([C4, output], axis=1))
        output = self.cls(output)
        if self.enable_auxiliary_loss:
            auxout = self.aux(C3)
            return [output, auxout]
        else:
            return [output]


class EMAU(nn.Layer):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
        momentum (float): The parameter for updating bases.
    '''

    def __init__(self, c, k, stage_num=3, momentum=0.1):
        super(EMAU, self).__init__()
        assert stage_num >= 1
        self.stage_num = stage_num
        self.momentum = momentum
        self.c = c

        tmp_mu = self.create_parameter(
            shape=[1, c, k],
            default_initializer=paddle.nn.initializer.KaimingNormal(k))
        mu = F.normalize(paddle.to_tensor(tmp_mu), axis=1, p=2)
        self.register_buffer('mu', mu)

    def forward(self, x):
        x_shape = paddle.shape(x)
        x = x.flatten(2)
        mu = paddle.tile(self.mu, [x_shape[0], 1, 1])

        with paddle.no_grad():
            for i in range(self.stage_num):
                x_t = paddle.transpose(x, [0, 2, 1])
                z = paddle.bmm(x_t, mu)
                z = F.softmax(z, axis=2)
                z_ = F.normalize(z, axis=1, p=1)
                mu = paddle.bmm(x, z_)
                mu = F.normalize(mu, axis=1, p=2)

        z_t = paddle.transpose(z, [0, 2, 1])
        x = paddle.matmul(mu, z_t)
        x = paddle.reshape(x, [0, self.c, x_shape[2], x_shape[3]])

        if self.training:
            mu = paddle.mean(mu, 0, keepdim=True)
            mu = F.normalize(mu, axis=1, p=2)
            mu = self.mu * (1 - self.momentum) + mu * self.momentum
            if paddle.distributed.get_world_size() > 1:
                mu = paddle.distributed.all_reduce(mu)
                mu /= paddle.distributed.get_world_size()
            self.mu = mu

        return x
