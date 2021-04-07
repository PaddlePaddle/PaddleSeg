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
class DNLNet(nn.Layer):
    """Disentangled Non-Local Neural Networks.

    The original article refers to
    Minghao Yin, et al. "Disentangled Non-Local Neural Networks"
    (https://arxiv.org/abs/2006.06668)
    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of output of backbone.
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: False.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian'.
        temperature (float): Temperature to adjust attention. Default: 0.05.
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
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 temperature=0.05,
                 concat_input=True,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.head = DNLHead(num_classes, in_channels, reduction, use_scale,
                            mode, temperature, concat_input,
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
                align_corners=self.align_corners,
                align_mode=1) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DNLHead(nn.Layer):
    """
    The DNLNet head.

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (tuple): The number of input channels.
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: False.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
        temperature (float): Temperature to adjust attention. Default: 0.05
        concat_input (bool): Whether concat the input and output of convs before classification layer. Default: True
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 reduction,
                 use_scale,
                 mode,
                 temperature,
                 concat_input=True,
                 enable_auxiliary_loss=True,
                 **kwargs):
        super(DNLHead, self).__init__()
        self.in_channels = in_channels[-1]
        self.concat_input = concat_input
        self.enable_auxiliary_loss = enable_auxiliary_loss
        inter_channels = self.in_channels // 4

        self.dnl_block = DisentangledNonLocal2D(
            in_channels=inter_channels,
            reduction=reduction,
            use_scale=use_scale,
            temperature=temperature,
            mode=mode)
        self.conv0 = layers.ConvBNReLU(
            in_channels=self.in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            bias_attr=False)
        self.conv1 = layers.ConvBNReLU(
            in_channels=inter_channels,
            out_channels=inter_channels,
            kernel_size=3,
            bias_attr=False)
        self.cls = nn.Sequential(
            nn.Dropout2D(p=0.1), nn.Conv2D(inter_channels, num_classes, 1))
        self.aux = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=1024,
                out_channels=256,
                kernel_size=3,
                bias_attr=False), nn.Dropout2D(p=0.1),
            nn.Conv2D(256, num_classes, 1))
        if self.concat_input:
            self.conv_cat = layers.ConvBNReLU(
                self.in_channels + inter_channels,
                inter_channels,
                kernel_size=3,
                bias_attr=False)

    def forward(self, feat_list):
        C3, C4 = feat_list
        output = self.conv0(C4)
        output = self.dnl_block(output)
        output = self.conv1(output)
        if self.concat_input:
            output = self.conv_cat(paddle.concat([C4, output], axis=1))
        output = self.cls(output)
        if self.enable_auxiliary_loss:
            auxout = self.aux(C3)
            return [output, auxout]
        else:
            return [output]


class DisentangledNonLocal2D(layers.NonLocal2D):
    """Disentangled Non-Local Blocks.

    Args:
        temperature (float): Temperature to adjust attention.
    """

    def __init__(self, temperature, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2D(self.in_channels, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = paddle.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight /= self.temperature
        pairwise_weight = F.softmax(pairwise_weight, -1)
        return pairwise_weight

    def forward(self, x):
        x_shape = paddle.shape(x)
        g_x = self.g(x).reshape([0, self.inter_channels,
                                 -1]).transpose([0, 2, 1])

        if self.mode == "gaussian":
            theta_x = paddle.transpose(
                x.reshape([0, self.in_channels, -1]), [0, 2, 1])
            if self.sub_sample:
                phi_x = paddle.transpose(self.phi(x), [0, self.in_channels, -1])
            else:
                phi_x = paddle.transpose(x, [0, self.in_channels, -1])

        elif self.mode == "concatenation":
            theta_x = paddle.reshape(
                self.theta(x), [0, self.inter_channels, -1, 1])
            phi_x = paddle.reshape(self.phi(x), [0, self.inter_channels, 1, -1])

        else:
            theta_x = self.theta(x).reshape([0, self.inter_channels,
                                             -1]).transpose([0, 2, 1])
            phi_x = paddle.reshape(self.phi(x), [0, self.inter_channels, -1])

        theta_x -= paddle.mean(theta_x, axis=-2, keepdim=True)
        phi_x -= paddle.mean(phi_x, axis=-1, keepdim=True)

        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)

        y = paddle.matmul(pairwise_weight, g_x).transpose([0, 2, 1]).reshape(
            [0, self.inter_channels, x_shape[2], x_shape[3]])
        unary_mask = F.softmax(
            paddle.reshape(self.conv_mask(x), [0, 1, -1]), -1)
        unary_x = paddle.matmul(unary_mask, g_x).transpose([0, 2, 1]).reshape(
            [0, self.inter_channels, 1, 1])
        output = x + self.conv_out(y + unary_x)
        return output
