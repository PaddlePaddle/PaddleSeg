# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn.functional as F
from paddle import nn

from paddleseg.cvlibs import manager
from paddleseg.models.common import pyramid_pool
from paddleseg.models.common.layer_libs import ConvBNReLU, SeparableConvBNReLU, AuxLayer
from paddleseg.utils import utils


@manager.MODELS.add_component
class FastSCNN(nn.Layer):
    """
    The FastSCNN implementation based on PaddlePaddle.

    As mentioned in the original paper, FastSCNN is a real-time segmentation algorithm (123.5fps)
    even for high resolution images (1024x2048).

    The original article refers to
        Poudel, Rudra PK, et al. "Fast-scnn: Fast semantic segmentation network."
        (https://arxiv.org/pdf/1902.04502.pdf)

    Args:

        num_classes (int): the unique number of target classes. Default to 2.
        enable_auxiliary_loss (bool): a bool values indicates whether adding auxiliary loss.
            if true, auxiliary loss will be added after LearningToDownsample module, where the weight is 0.4. Default to False.
        pretrained (str): the path of pretrained model. Default to None.
    """

    def __init__(self, num_classes, enable_auxiliary_loss=True,
                 pretrained=None):

        super(FastSCNN, self).__init__()

        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(
            64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifier(128, num_classes)

        if enable_auxiliary_loss:
            self.auxlayer = AuxLayer(64, 32, num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.init_weight()
        utils.load_entire_model(self, pretrained)

    def forward(self, input, label=None):
        logit_list = []
        higher_res_features = self.learning_to_downsample(input)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        logit = self.classifier(x)
        logit = F.resize_bilinear(logit, input.shape[2:])
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_logit = self.auxlayer(higher_res_features)
            auxiliary_logit = F.resize_bilinear(auxiliary_logit,
                                                input.shape[2:])
            logit_list.append(auxiliary_logit)

        return logit_list

    def init_weight(self):
        """
        Initialize the parameters of model parts.
        """
        pass


class LearningToDownsample(nn.Layer):
    """
    Learning to downsample module.

    This module consists of three downsampling blocks (one Conv and two separable Conv)

    Args:
        dw_channels1 (int): the input channels of the first sep conv. Default to 32.
        dw_channels2 (int): the input channels of the second sep conv. Default to 48.
        out_channels (int): the output channels of LearningToDownsample module. Default to 64.
    """

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=3, out_channels=dw_channels1, kernel_size=3, stride=2)
        self.dsconv_bn_relu1 = SeparableConvBNReLU(
            in_channels=dw_channels1,
            out_channels=dw_channels2,
            kernel_size=3,
            stride=2,
            padding=1)
        self.dsconv_bn_relu2 = SeparableConvBNReLU(
            in_channels=dw_channels2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dsconv_bn_relu1(x)
        x = self.dsconv_bn_relu2(x)
        return x


class GlobalFeatureExtractor(nn.Layer):
    """
    Global feature extractor module

    This module consists of three LinearBottleneck blocks (like inverted residual introduced by MobileNetV2) and
    a PPModule (introduced by PSPNet).

    Args:
        in_channels (int): the number of input channels to the module. Default to 64.
        block_channels (tuple): a tuple represents output channels of each bottleneck block. Default to (64, 96, 128).
        out_channels (int): the number of output channels of the module. Default to 128.
        expansion (int): the expansion factor in bottleneck. Default to 6.
        num_blocks (tuple): it indicates the repeat time of each bottleneck. Default to (3, 3, 3).
    """

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 expansion=6,
                 num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()

        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels,
                                            block_channels[0], num_blocks[0],
                                            expansion, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0],
                                            block_channels[1], num_blocks[1],
                                            expansion, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1],
                                            block_channels[2], num_blocks[2],
                                            expansion, 1)

        self.ppm = pyramid_pool.PPModule(
            block_channels[2], out_channels, dim_reduction=True)

    def _make_layer(self,
                    block,
                    in_channels,
                    out_channels,
                    blocks,
                    expansion=6,
                    stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, expansion, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, expansion, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class LinearBottleneck(nn.Layer):
    """
    Single bottleneck implementation.

    Args:
        in_channels (int): the number of input channels to bottleneck block.
        out_channels (int): the number of output channels of bottleneck block.
        expansion (int). the expansion factor in bottleneck. Default to 6.
        stride (int). the stride used in depth-wise conv.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=6,
                 stride=2,
                 **kwargs):
        super(LinearBottleneck, self).__init__()

        self.use_shortcut = stride == 1 and in_channels == out_channels

        expand_channels = in_channels * expansion
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=expand_channels,
                kernel_size=1,
                bias_attr=False),
            # dw
            ConvBNReLU(
                in_channels=expand_channels,
                out_channels=expand_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expand_channels,
                bias_attr=False),
            # pw-linear
            nn.Conv2d(
                in_channels=expand_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False),
            nn.SyncBatchNorm(out_channels))

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class FeatureFusionModule(nn.Layer):
    """
    Feature Fusion Module Implementation.

    This module fuses high-resolution feature and low-resolution feature.

    Args:
        high_in_channels (int): the channels of high-resolution feature (output of LearningToDownsample).
        low_in_channels (int). the channels of low-resolution feature (output of GlobalFeatureExtractor).
        out_channels (int). the output channels of this module.
    """

    def __init__(self, high_in_channels, low_in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        # There only depth-wise conv is used WITHOUT point-wise conv
        self.dwconv = ConvBNReLU(
            in_channels=low_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=128,
            bias_attr=False)

        self.conv_low_res = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1), nn.SyncBatchNorm(out_channels))

        self.conv_high_res = nn.Sequential(
            nn.Conv2d(
                in_channels=high_in_channels,
                out_channels=out_channels,
                kernel_size=1), nn.SyncBatchNorm(out_channels))

        self.relu = nn.ReLU(True)

    def forward(self, high_res_input, low_res_input):
        low_res_input = F.resize_bilinear(input=low_res_input, scale=4)
        low_res_input = self.dwconv(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)
        high_res_input = self.conv_high_res(high_res_input)
        x = high_res_input + low_res_input

        return self.relu(x)


class Classifier(nn.Layer):
    """
    The Classifier module implementation.

    This module consists of two depth-wise conv and one conv.

    Args:
        input_channels (int): the input channels to this module.
        num_classes (int). the unique number of target classes.
    """

    def __init__(self, input_channels, num_classes):
        super(Classifier, self).__init__()

        self.dsconv1 = SeparableConvBNReLU(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1)

        self.dsconv2 = SeparableConvBNReLU(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1)

        self.conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = F.dropout(x, p=0.1)  # dropout_prob
        x = self.conv(x)
        return x
