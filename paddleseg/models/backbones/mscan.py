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

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign, Normal

from paddleseg.cvlibs import manager
from paddleseg.models.backbones.transformer_utils import (DropPath, ones_,
                                                          to_2tuple, zeros_)
from paddleseg.models.layers import SyncBatchNorm
from paddleseg.utils import utils

__all__ = ["MSCAN", "MSCAN_T", "MSCAN_S", "MSCAN_B", "MSCAN_L"]


def get_depthwise_conv(dim, kernel_size=3):
    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    padding = tuple([k // 2 for k in kernel_size])
    return Conv2D(
        dim, dim, kernel_size, padding=padding, bias_attr=True, groups=dim)


class Mlp(nn.Layer):
    """Multilayer perceptron."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = get_depthwise_conv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels, ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            SyncBatchNorm(out_channels // 2),
            nn.GELU(),
            nn.Conv2D(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            SyncBatchNorm(out_channels))

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose([0, 2, 1])
        return x, H, W


class AttentionModule(nn.Layer):
    """
    AttentionModule Layer, which contains some depth-wise strip convolutions.

    Args:
        dim (int): Number of input channels.
        kernel_sizes (list[int], optional): The height or width of each strip convolution kernel. Default: [7, 11, 21].
    """

    def __init__(self, dim, kernel_sizes=[7, 11, 21]):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)

        self.dwconvs = nn.LayerList([
            nn.Sequential((f"conv{i+1}_1", get_depthwise_conv(dim, (1, k))),
                          (f"conv{i+1}_2", get_depthwise_conv(dim, (k, 1))))
            for i, k in enumerate(kernel_sizes)
        ])

        self.conv_out = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        u = paddle.clone(x)
        attn = self.conv0(x)

        attns = [m(attn) for m in self.dwconvs]

        attn += sum(attns)

        attn = self.conv_out(attn)

        return attn * u


class SpatialAttention(nn.Layer):
    """
    SpatialAttention Layer.

    Args:
        d_model (int): Number of input channels.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
    """

    def __init__(self, d_model, atten_kernel_sizes=[7, 11, 21]):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2D(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, atten_kernel_sizes)
        self.proj_2 = nn.Conv2D(d_model, d_model, 1)

    def forward(self, x):
        shorcut = paddle.clone(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Layer):
    """
    MSCAN Block.

    Args:
        dim (int): Number of feature channels.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU.
    """

    def __init__(
            self,
            dim,
            atten_kernel_sizes=[7, 11, 21],
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU, ):
        super().__init__()
        self.norm1 = SyncBatchNorm(dim)
        self.attn = SpatialAttention(dim, atten_kernel_sizes)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = SyncBatchNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        layer_scale_init_value = paddle.full(
            [dim, 1, 1], fill_value=1e-2, dtype="float32")
        self.layer_scale_1 = paddle.create_parameter(
            [dim, 1, 1], "float32", attr=Assign(layer_scale_init_value))
        self.layer_scale_2 = paddle.create_parameter(
            [dim, 1, 1], "float32", attr=Assign(layer_scale_init_value))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        x = x.reshape([B, C, N]).transpose([0, 2, 1])
        return x


class OverlapPatchEmbed(nn.Layer):
    """
    An Opverlaping Image to Patch Embedding Layer.

    Args:
        patch_size (int, optional): Patch token size. Default: 7.
        stride (int, optional): Stride of Convolution in OverlapPatchEmbed. Default: 4.
        in_chans (int, optional): Number of input image channels. Default: 3.
        embed_dim (int, optional): Number of linear projection output channels. Default: 768.
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = SyncBatchNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = self.norm(x)

        x = x.flatten(2).transpose([0, 2, 1])
        return x, H, W


def _check_length(*args):
    target_length = len(args[0])
    for item in args:
        if target_length != len(item):
            return False
    return True


@manager.BACKBONES.add_component
class MSCAN(nn.Layer):
    """
    The MSCAN implementation based on PaddlePaddle.

    The original article refers to
    Guo, Meng-Hao, et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
    (https://arxiv.org/pdf/2209.08575.pdf)

    Args:
        in_channels (int, optional): Number of input image channels. Default: 3.
        embed_dims (list[int], optional): Number of each stage output channels. Default: [32, 64, 160, 256].
        depths (list[int], optional): Depths of each MSCAN stage.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float, optional): Dropout rate. Default: 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[32, 64, 160, 256],
                 depths=[3, 3, 5, 2],
                 mlp_ratios=[8, 8, 4, 4],
                 atten_kernel_sizes=[7, 11, 21],
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 pretrained=None):
        super().__init__()
        if not _check_length(embed_dims, mlp_ratios, depths):
            raise ValueError(
                "The length of aurgments 'embed_dims', 'mlp_ratios' and 'drop_path_rate' must be same."
            )

        self.depths = depths
        self.num_stages = len(embed_dims)
        self.feat_channels = embed_dims

        drop_path_rates = [
            x for x in np.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.LayerList([
                Block(
                    dim=embed_dims[i],
                    atten_kernel_sizes=atten_kernel_sizes,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=drop_path_rates[cur + j])
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            utils.load_pretrained_model(self, pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.LayerNorm):
                    zeros_(sublayer.bias)
                    ones_(sublayer.weight)
                elif isinstance(sublayer, nn.Conv2D):
                    fan_out = (sublayer._kernel_size[0] *
                               sublayer._kernel_size[1] *
                               sublayer._out_channels)
                    fan_out //= sublayer._groups
                    initializer = Normal(mean=0, std=math.sqrt(2.0 / fan_out))
                    initializer(sublayer.weight)
                    zeros_(sublayer.bias)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
            outs.append(x)

        return outs


@manager.BACKBONES.add_component
def MSCAN_T(**kwargs):
    return MSCAN(**kwargs)


@manager.BACKBONES.add_component
def MSCAN_S(**kwargs):
    return MSCAN(embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2], **kwargs)


@manager.BACKBONES.add_component
def MSCAN_B(**kwargs):
    return MSCAN(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        drop_path_rate=0.1,
        **kwargs)


@manager.BACKBONES.add_component
def MSCAN_L(**kwargs):
    return MSCAN(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        drop_path_rate=0.3,
        **kwargs)
