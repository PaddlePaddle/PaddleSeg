# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/microsoft/Swin-Transformer
# reference: https://arxiv.org/abs/2103.14030

import numpy as np
import paddle
import paddle.nn as nn

from medicalseg.cvlibs import manager, param_init
from medicalseg.utils import load_pretrained_model
from medicalseg.models.backbones.transformer_utils import DropPath, to_2tuple

MODEL_URLS = {
    "SwinTransformer_tiny_patch4_window7_224":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_tiny_patch4_window7_224_pretrained/model.pdparams",
    "SwinTransformer_small_patch4_window7_224":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_small_patch4_window7_224_pretrained/model.pdparams",
    "SwinTransformer_base_patch4_window7_224":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_base_patch4_window7_224_pretrained/model.pdparams",
    "SwinTransformer_base_patch4_window12_384":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_base_patch4_window12_384_pretrained/model.pdparams",
    "SwinTransformer_large_patch4_window7_224":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained/model.pdparams",
    "SwinTransformer_large_patch4_window12_384":
    "https://paddleseg.bj.bcebos.com/paddleseg3d/backbone/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained/model.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        C (int): Channels of image
    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default query and key scale of head_dim ** -0.5 if set. Default None
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                   num_heads),
            default_initializer=nn.initializer.Constant(value=0.))
        self.add_parameter("relative_position_bias_table",
                           self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2

        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[
            0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        param_init.trunc_normal(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def eval(self):
        super().eval()
        # this is used to re-param swin for model export
        relative_position_bias_table = self.relative_position_bias_table
        window_size = self.window_size
        index = self.relative_position_index.reshape([-1])

        relative_position_bias = paddle.index_select(
            relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape([
            window_size[0] * window_size[1], window_size[0] * window_size[1], -1
        ])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.unsqueeze(0)
        self.register_buffer("relative_position_bias", relative_position_bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            [B_, N, 3, self.num_heads, C // self.num_heads]).transpose(
                [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        if self.training or not hasattr(self, "relative_position_bias"):
            index = self.relative_position_index.reshape([-1])

            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index)
            relative_position_bias = relative_position_bias.reshape([
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1
            ])  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + self.relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N
                                 ]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int, optional): Window size. Default: 7
        shift_size (int, optional): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default query and key scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype("float32")
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size,
             C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W,
                                   C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size ({}*{}) are not even.".format(
            H, W)

        x = x.reshape([B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, H * W // 4, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default query and key scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    Args:
        img_size (int, optional): Image size.  Default: 224.
        patch_size (int, optional): Patch token size. Default: 4.
        in_chans (int, optional): Number of input image channels. Default: 3.
        embed_dim (int, optional): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


@manager.BACKBONES.add_component
class SwinTransformer(nn.Layer):
    """ Swin Transformer
        A PaddlePaddle implementation of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple[int], optional): Input image size. Default 224
        patch_size (int | tuple[int], optional): Patch size. Default: 4
        in_chans (int, optional): Number of input image channels. Default: 3
        num_classes (int, optional): Number of classes for classification head. Default: 1000
        embed_dim (int, optional): Patch embedding dimension. Default: 96
        depths (tuple[int], optional): Depth of each Swin Transformer layer.
        num_heads (tuple[int], optional): Number of attention heads in different layers.
        window_size (int, optional): Window size. Default: 7
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float, optional): Override default query and key scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float, optional): Dropout rate. Default: 0
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm.
        ape (bool, optional): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool, optional): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim),
                default_initializer=nn.initializer.Constant(value=0.))
            self.add_parameter("absolute_pos_embed", self.absolute_pos_embed)
            param_init.trunc_normal(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        #self.avgpool = nn.AdaptiveAvgPool1D(1)
        #self.head = nn.Linear(
        #self.num_features,
        #num_classes) if self.num_classes > 0 else nn.Identity()

        self.init_weight()

    def init_weight(self):
        """Initialize the parameters of model parts."""
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Linear):
                param_init.trunc_normal(sublayer.weight)
                if sublayer.bias is not None:
                    param_init.constant_init(sublayer.bias, value=0.0)
            elif isinstance(sublayer, nn.LayerNorm):
                param_init.constant_init(sublayer.bias, value=0.0)
                param_init.constant_init(sublayer.weight, value=1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
        #x = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        #x = paddle.flatten(x, 1)
        return x, x_downsample

    def forward(self, x):
        out = self.forward_features(x)
        # x = = self.forward_features(x)
        # x = self.head(x)
        return out


@manager.BACKBONES.add_component
def SwinTransformer_tinier_patch4_window7_224(pretrained=False, **kwargs):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def SwinTransformer_tiny_patch4_window7_224(pretrained=False, **kwargs):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_tiny_patch4_window7_224"])
    return model


@manager.BACKBONES.add_component
def SwinTransformer_small_patch4_window7_224(pretrained=False, **kwargs):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_small_patch4_window7_224"])
    return model


@manager.BACKBONES.add_component
def SwinTransformer_base_patch4_window7_224(pretrained=False, **kwargs):
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_base_patch4_window7_224"])
    return model


@manager.BACKBONES.add_component
def SwinTransformer_base_patch4_window12_384(pretrained=False, **kwargs):
    model = SwinTransformer(
        img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.5,  # NOTE: do not appear in offical code
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_base_patch4_window12_384"])
    return model


@manager.BACKBONES.add_component
def SwinTransformer_large_patch4_window7_224(pretrained=False, **kwargs):
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_large_patch4_window7_224"])
    return model


@manager.BACKBONES.add_component
def SwinTransformer_large_patch4_window12_384(pretrained=False, **kwargs):
    model = SwinTransformer(
        img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        **kwargs)
    if pretrained:
        load_pretrained_model(
            model, MODEL_URLS["SwinTransformer_large_patch4_window12_384"])
    return model
