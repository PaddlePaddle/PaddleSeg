# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
import math
from typing import List, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal, Uniform
from paddle import Tensor

from paddleseg.cvlibs import manager
from paddleseg.models.layers import ConvBNAct

__all__ = [
    "PIDNet_Small",
    "PIDNet_Medium",
    "PIDNet_Large",
]


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv_bn_relu = ConvBNAct(inplanes,
                                      planes,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      act_type='relu',
                                      bias_attr=False)
        self.relu = nn.ReLU()
        self.conv_bn = ConvBNAct(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias_attr=False)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv_bn_relu1 = ConvBNAct(inplanes,
                                       planes,
                                       kernel_size=1,
                                       act_type='relu',
                                       bias_attr=False)
        self.conv_bn_relu2 = ConvBNAct(planes,
                                       planes,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       act_type='relu',
                                       bias_attr=False)
        self.conv_bn = ConvBNAct(planes,
                                 planes * self.expansion,
                                 kernel_size=1,
                                 bias_attr=False)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Layer):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels

        self.scales = nn.LayerList([
            nn.Sequential(
                nn.SyncBatchNorm(in_channels), nn.ReLU(),
                nn.Conv2D(in_channels,
                          branch_channels,
                          kernel_size=1,
                          bias_attr=False))
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                nn.Sequential(*[
                    nn.AvgPool2D(kernel_size=kernel_sizes[i - 1],
                                 stride=strides[i - 1],
                                 padding=paddings[i - 1],
                                 exclusive=False),
                    nn.Sequential(
                        nn.SyncBatchNorm(in_channels), nn.ReLU(),
                        nn.Conv2D(in_channels,
                                  branch_channels,
                                  kernel_size=1,
                                  bias_attr=False))
                ]))
        self.scales.append(
            nn.Sequential(*[
                nn.AdaptiveAvgPool2D((1, 1)),
                nn.Sequential(
                    nn.SyncBatchNorm(in_channels), nn.ReLU(),
                    nn.Conv2D(in_channels,
                              branch_channels,
                              kernel_size=1,
                              bias_attr=False))
            ]))
        self.processes = nn.LayerList()
        for i in range(num_scales - 1):
            self.processes.append(
                nn.Sequential(
                    nn.SyncBatchNorm(branch_channels), nn.ReLU(),
                    nn.Conv2D(branch_channels,
                              branch_channels,
                              kernel_size=3,
                              padding=1,
                              bias_attr=False)))

        self.compression = nn.Sequential(
            nn.SyncBatchNorm(branch_channels * num_scales), nn.ReLU(),
            nn.Conv2D(branch_channels * num_scales,
                      out_channels,
                      kernel_size=1,
                      bias_attr=False))

        self.shortcut = nn.Sequential(
            nn.SyncBatchNorm(in_channels), nn.ReLU(),
            nn.Conv2D(in_channels, out_channels, kernel_size=1,
                      bias_attr=False))

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(self.scales[i](inputs),
                                    size=inputs.shape[2:],
                                    mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(paddle.concat(feats,
                                              axis=1)) + self.shortcut(inputs)


class PAPPM(DAPPM):
    """PAPPM module in `PIDNet <https://arxiv.org/abs/2206.02066>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels, num_scales,
                         kernel_sizes, strides, paddings, upsample_mode)

        self.processes = nn.Sequential(
            nn.SyncBatchNorm(self.branch_channels * (self.num_scales - 1)),
            nn.ReLU(),
            nn.Conv2D(self.branch_channels * (self.num_scales - 1),
                      self.branch_channels * (self.num_scales - 1),
                      kernel_size=3,
                      padding=1,
                      groups=self.num_scales - 1,
                      bias_attr=False))

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(self.scales[i](inputs),
                                    size=inputs.shape[2:],
                                    mode=self.unsample_mode,
                                    align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(paddle.concat(feats, axis=1))
        return self.compression(paddle.concat([x_, scale_out],
                                              axis=1)) + self.shortcut(inputs)


class PagFM(nn.Layer):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear'):
        super().__init__()
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.f_i = ConvBNAct(in_channels, channels, 1, bias_attr=False)
        self.f_p = ConvBNAct(in_channels, channels, 1, bias_attr=False)
        if with_channel:
            self.up = ConvBNAct(channels, in_channels, 1, bias_attr=False)
        if after_relu:
            self.relu = nn.ReLU()

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(f_i,
                            size=x_p.shape[2:],
                            mode=self.upsample_mode,
                            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = F.sigmoid(self.up(f_p * f_i))
        else:
            sigma = F.sigmoid(paddle.sum(f_p * f_i, axis=1).unsqueeze(1))

        x_i = F.interpolate(x_i,
                            size=x_p.shape[2:],
                            mode=self.upsample_mode,
                            align_corners=False)

        out = sigma * x_i + (1 - sigma) * x_p
        return out


class Bag(nn.Layer):
    """Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolution. Default: 3.
        padding (int): The padding of the convolution. Default: 1.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.SyncBatchNorm(in_channels), nn.ReLU(),
            nn.Conv2D(in_channels,
                      out_channels,
                      kernel_size,
                      padding=padding,
                      bias_attr=False))

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with boundary-attention-guided fusion.
        """
        sigma = F.sigmoid(x_d)
        return self.conv(sigma * x_p + (1 - sigma) * x_i)


class LightBag(nn.Layer):
    """Light Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.f_p = ConvBNAct(in_channels,
                             out_channels,
                             kernel_size=1,
                             bias_attr=False)
        self.f_i = ConvBNAct(in_channels,
                             out_channels,
                             kernel_size=1,
                             bias_attr=False)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.
        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with light boundary-attention-guided
                fusion.
        """
        sigma = F.sigmoid(x_d)

        f_p = self.f_p((1 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)

        return f_p + f_i


class PIDNet(nn.Layer):
    """PIDNet backbone.

    This backbone is the implementation of `PIDNet: A Real-time Semantic
    Segmentation Network Inspired from PID Controller
    <https://arxiv.org/abs/2206.02066>`_.
    Modified from https://github.com/XuJiacong/PIDNet.

    Licensed under the MIT License.

    Args:
        in_channels (int): The number of input channels. Default: 3.
        channels (int): The number of channels in the stem layer. Default: 64.
        ppm_channels (int): The number of channels in the PPM layer.
            Default: 96.
        num_stem_blocks (int): The number of blocks in the stem layer.
            Default: 2.
        num_branch_blocks (int): The number of blocks in the branch layer.
            Default: 3.
        align_corners (bool): The align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.LayerList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(block=BasicBlock if i < 2 else Bottleneck,
                                 in_channels=channels * 2**(i + 1),
                                 channels=channels * 8 if i > 0 else channels *
                                 4,
                                 num_blocks=num_branch_blocks if i < 2 else 2,
                                 stride=2))

        # P Branch
        self.p_branch_layers = nn.LayerList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(block=BasicBlock if i < 2 else Bottleneck,
                                 in_channels=channels * 2,
                                 channels=channels * 2,
                                 num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvBNAct(channels * 4,
                                       channels * 2,
                                       kernel_size=1,
                                       bias_attr=False)
        self.compression_2 = ConvBNAct(channels * 8,
                                       channels * 2,
                                       kernel_size=1,
                                       bias_attr=False)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.LayerList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
            self.spp = PAPPM(channels * 16,
                             ppm_channels,
                             channels * 4,
                             num_scales=5)
            self.dfm = LightBag(channels * 4, channels * 4)
        else:
            self.d_branch_layers = nn.LayerList([
                self._make_single_layer(BasicBlock, channels * 2, channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
            self.spp = DAPPM(channels * 16,
                             ppm_channels,
                             channels * 4,
                             num_scales=5)
            self.dfm = Bag(channels * 4, channels * 4)

        self.diff_1 = ConvBNAct(channels * 4,
                                channels * channel_expand,
                                kernel_size=3,
                                padding=1,
                                bias_attr=False)
        self.diff_2 = ConvBNAct(channels * 8,
                                channels * 2,
                                kernel_size=3,
                                padding=1,
                                bias_attr=False)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

        # for channels of four returned stages
        self.feat_channels = [channels * 4]

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                fan_out = layer.weight.shape[0] * \
                          layer.weight.shape[2] * layer.weight.shape[3]
                std = math.sqrt(2) / math.sqrt(fan_out)
                Normal(0, std)(layer.weight)
                if layer.bias is not None:
                    fan_in = layer.weight.shape[1] * \
                             layer.weight.shape[2] * layer.weight.shape[3]
                    bound = 1 / math.sqrt(fan_in)
                    Uniform(-bound, bound)(layer.bias)
            elif isinstance(layer, (nn.BatchNorm2D, nn.SyncBatchNorm)):
                Constant(1)(layer.weight)
                Constant(0)(layer.bias)

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:
            nn.Sequential: The stem layer.
        """

        layers = [
            ConvBNAct(in_channels,
                      channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      act_type='relu',
                      bias_attr=False),
            ConvBNAct(channels,
                      channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      act_type='relu',
                      bias_attr=False),
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(BasicBlock,
                             channels,
                             channels * 2,
                             num_blocks,
                             stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvBNAct(in_channels,
                                   channels * block.expansion,
                                   kernel_size=1,
                                   stride=stride,
                                   bias_attr=False)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(in_channels,
                      channels,
                      stride=1,
                      no_relu=(i == num_blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Layer:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Layer
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvBNAct(in_channels,
                                   channels * block.expansion,
                                   kernel_size=1,
                                   stride=stride,
                                   bias_attr=False)
        return block(in_channels, channels, stride, downsample, no_relu=True)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(diff_i,
                             size=[h_out, w_out],
                             mode='bilinear',
                             align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(diff_i,
                             size=[h_out, w_out],
                             mode='bilinear',
                             align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))

        x_i = self.spp(x_i)
        x_i = F.interpolate(x_i,
                            size=[h_out, w_out],
                            mode='bilinear',
                            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out


@manager.BACKBONES.add_component
def PIDNet_Small(**kwargs):
    model = PIDNet(channels=32,
                   ppm_channels=96,
                   num_stem_blocks=2,
                   num_branch_blocks=3,
                   align_corners=False,
                   **kwargs)
    return model


@manager.BACKBONES.add_component
def PIDNet_Medium(**kwargs):
    model = PIDNet(channels=64,
                   ppm_channels=96,
                   num_stem_blocks=2,
                   num_branch_blocks=3,
                   align_corners=False,
                   **kwargs)
    return model


@manager.BACKBONES.add_component
def PIDNet_Large(**kwargs):
    model = PIDNet(channels=64,
                   ppm_channels=112,
                   num_stem_blocks=3,
                   num_branch_blocks=4,
                   align_corners=False,
                   **kwargs)
    return model
