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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.backbones.transformer_utils import Identity


@manager.MODELS.add_component
class UPerNetCAE(nn.Layer):
    """ 
    The UPerNet  with CAE as backbone implementation with PaddlePaddle.

    The original paper refers to Unified Perceptual Parsing for Scene Understanding.
    (https://arxiv.org/abs/1807.10221)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        channels(int): Hidden layer channels of upernet head.
        fpn_channels(int): The fpn_channels of upernet head.
        head_channels(int): The inplane of upernet head.
        channels_fpn(int): The channels_fpn of upernet head.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        dropout_ratio(float): The dropout ratio of upernet head.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 channels,
                 fpn_channels,
                 head_channels,
                 channels_fpn,
                 enable_auxiliary_loss=True,
                 align_corners=True,
                 dropout_ratio=0.1,
                 pretrained=None):
        super(UPerNetCAE, self).__init__()
        self._init_fpn(embed_dim=768, patch_size=16)
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.fpn_channels = fpn_channels
        self.head_channels = head_channels
        self.channels_fpn = channels_fpn

        self.decode_head = UPerNetHead(
            inplane=head_channels,
            num_class=num_classes,
            channels_fpn=channels_fpn,
            dropout_ratio=dropout_ratio,
            channels=channels,
            fpn_channels=fpn_channels,
            enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def _init_fpn(self, embed_dim=768, patch_size=16, out_with_norm=False):
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(embed_dim,
                                   embed_dim,
                                   kernel_size=2,
                                   stride=2),
                nn.SyncBatchNorm(embed_dim, momentum=0.1),
                nn.GELU(),
                nn.Conv2DTranspose(embed_dim,
                                   embed_dim,
                                   kernel_size=2,
                                   stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.Conv2DTranspose(embed_dim,
                                   embed_dim,
                                   kernel_size=2,
                                   stride=2), )

            self.fpn3 = Identity()

            self.fpn4 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(embed_dim,
                                   embed_dim,
                                   kernel_size=2,
                                   stride=2), )

            self.fpn2 = Identity()

            self.fpn3 = nn.Sequential(nn.MaxPool2D(kernel_size=2, stride=2), )

            self.fpn4 = nn.Sequential(nn.MaxPool2D(kernel_size=4, stride=4), )

        if not out_with_norm:
            self.norm = Identity()
        else:
            self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)

    def forward(self, x):
        feats, feats_shape = self.backbone(x)  # [1, 1024, 768]
        B, _, Hp, Wp = feats_shape

        feats = [feats[i] for i in self.backbone_indices]

        for i, feat in enumerate(feats):
            feats[i] = paddle.reshape(paddle.transpose(self.norm(feat),
                                                       perm=[0, 2, 1]),
                                      shape=[B, -1, Hp, Wp])

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(feats)):
            feats[i] = ops[i](feats[i])

        logit_list = self.decode_head(feats)
        logit_list = [
            F.interpolate(logit,
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=False) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PPModuleCAE(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.

    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, in_channels, out_channels, bin_sizes, dim_reduction,
                 align_corners):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_bn_relu2 = layers.ConvBNReLU(in_channels=in_channels +
                                               inter_channels * len(bin_sizes),
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               padding=1,
                                               bias_attr=False)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.

        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.

        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = layers.ConvBNReLU(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 bias_attr=False)

        return nn.Sequential(prior, conv)

    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x,
                              input.shape[2:],
                              mode='bilinear',
                              align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out


class UPerNetHead(nn.Layer):
    """
    The UPerNetHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        channels_fpn (list): The feature channels from backbone.
        fpn_channels (int, optional): The input channels of FPN module. Default: 512.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """

    def __init__(self,
                 inplane,
                 num_class,
                 channels_fpn,
                 channels,
                 dropout_ratio=0.1,
                 fpn_channels=512,
                 enable_auxiliary_loss=False):
        super(UPerNetHead, self).__init__()
        self.psp_modules = PPModuleCAE(in_channels=inplane,
                                       out_channels=fpn_channels,
                                       bin_sizes=(1, 2, 3, 6),
                                       dim_reduction=False,
                                       align_corners=False)

        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.lateral_convs = []
        self.fpn_convs = []

        for fpn_inplane in channels_fpn[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2D(fpn_inplane, fpn_channels, 1, bias_attr=False),
                    layers.SyncBatchNorm(fpn_channels), nn.ReLU()))
            self.fpn_convs.append(
                nn.Sequential(
                    layers.ConvBNReLU(fpn_channels,
                                      fpn_channels,
                                      3,
                                      bias_attr=False)))

        self.lateral_convs = nn.LayerList(self.lateral_convs)
        self.fpn_convs = nn.LayerList(self.fpn_convs)

        if self.enable_auxiliary_loss:
            if dropout_ratio is not None:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(channels_fpn[2],
                                      256,
                                      3,
                                      padding=1,
                                      bias_attr=False),
                    nn.Dropout2D(dropout_ratio),
                    nn.Conv2D(256, num_class, kernel_size=1))
            else:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(channels_fpn[2],
                                      256,
                                      3,
                                      padding=1,
                                      bias_attr=False),
                    nn.Conv2D(256, num_class, kernel_size=1))

        if dropout_ratio is not None:
            self.dropout = nn.Dropout2D(dropout_ratio)
        else:
            self.dropout = None

        self.fpn_bottleneck = layers.ConvBNReLU(len(channels_fpn) * channels,
                                                channels,
                                                3,
                                                padding=1,
                                                bias_attr=False)
        self.conv_seg = nn.Conv2D(channels, num_class, kernel_size=1)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, conv_out):
        psp_out = self.psp_modules(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.lateral_convs[i](conv_x)
            prev_shape = conv_x.shape[2:]
            f = conv_x + F.interpolate(
                f, prev_shape, mode='bilinear', align_corners=False)
            fpn_feature_list.append(self.fpn_convs[i](f))

        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].shape[2:]
        # resize multi-scales feature
        for index in range(len(conv_out) - 1, 0, -1):
            fpn_feature_list[index] = F.interpolate(fpn_feature_list[index],
                                                    size=output_size,
                                                    mode='bilinear',
                                                    align_corners=False)
        fusion_out = paddle.concat(fpn_feature_list, 1)
        x = self.fpn_bottleneck(fusion_out)
        x = self.cls_seg(x)

        if self.enable_auxiliary_loss:
            dsn = self.dsn(conv_out[2])
            out.append(x)
            out.append(dsn)
            return out
        else:
            return [x]
