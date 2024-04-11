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
from medicalseg.models.backbones.swin_transformer import SwinTransformerBlock, window_partition
from medicalseg.utils import load_pretrained_model


@manager.MODELS.add_component
class SwinUNet(nn.Layer):
    r""" SwinUNet.
    Args:
        backbone: Backbone of SwinUnet
        image_size (int, optional): Image size of input. Default: 224
        path_size (int, optional): Patch size of Patch Embed. Default: 4
        in_chans (int, optional): The inchannel of input. Default: 3
        num_classes (int, optional): Number of class. Default: 1000
        embed_dim (int, optional): The embed dim. Default: 96
        depths (list, optional): Block number of every encoder layer. Default: [2, 2, 2, 2]
        depths_decoder (list, optional): Block number of every decoder layer. Default: [1, 2, 2, 2]
        num_heads (int, optional): Number of attention heads. Default: [3, 6, 12, 24]
        window_size (int, optional): Window size. Default: 7
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default query and key scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.0
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
        patch_norm (bool, optional): If True, add normalization after patch embedding. Default: True
        pretrained(bool |str, optional): The path of pre-trained model. Default: None
        final_unsample (bool, optional): The final sample class. Default: True
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[1, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 pretrained=None,
                 final_upsample=True,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.final_upsample = final_upsample

        patches_resolution = [img_size // patch_size] * 2
        self.patches_resolution = patches_resolution

        self.pretrained = pretrained

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule
        self.layers_up = nn.LayerList()
        self.concat_back_dim = nn.LayerList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                int(embed_dim *
                    2**(self.num_layers - 1 -
                        i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] //
                                      (2**(self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] //
                                      (2**(self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = BasicUpLayer(
                    dim=int(embed_dim * 2**(self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] //
                                      (2**(self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] //
                                      (2**(self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(
                        self.num_layers - 1 -
                        i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) +
                                              1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if
                    (i_layer < self.num_layers - 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample:
            self.up = FinalPatchExpandX4(
                input_resolution=(img_size // patch_size,
                                  img_size // patch_size),
                dim_scale=4,
                dim=embed_dim)
            self.output = nn.Conv2D(in_channels=embed_dim,
                                    out_channels=self.num_classes,
                                    kernel_size=1,
                                    bias_attr=False)

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

        if self.pretrained is not None:
            load_pretrained_model(self, self.pretrained)

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = paddle.concat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape[0:3]
        assert L == H * W, "input features has wrong size"

        if self.final_upsample:
            x = self.up(x)
            x = x.reshape((B, 4 * H, 4 * W, self.embed_dim))
            x = x.transpose((0, 3, 1, 2))  #B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        if self.training:
            x = paddle.squeeze(x, axis=2)
        else:
            x = paddle.squeeze(x, axis=0)
        x = x.tile([1, 3, 1, 1])

        x, x_downsample = self.backbone(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        x = paddle.unsqueeze(x, axis=2)

        return [x]

    def postprocess(self, logits, labels):
        logits = [logits.transpose((2, 1, 0, 3, 4))]
        labels = labels.squeeze(2)
        return logits, labels


class PatchExpand(nn.Layer):

    def __init__(self,
                 input_resolution,
                 dim,
                 dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias_attr=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        x = x.reshape((B, H, W, C))
        x = x.reshape((B, H, W, 2, 2, C // 4))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * 2, W * 2, C // 4))
        x = x.reshape((B, H * 2 * W * 2, C // 4))
        x = self.norm(x)

        return x


class FinalPatchExpandX4(nn.Layer):

    def __init__(self,
                 input_resolution,
                 dim,
                 dim_scale=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias_attr=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape[0:3]
        assert L == H * W, "input features has wrong size"

        x = x.reshape((B, H, W, C))
        x = x.reshape(
            (B, H, W, self.dim_scale, self.dim_scale, C // (self.dim_scale**2)))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * self.dim_scale, W * self.dim_scale,
                       C // (self.dim_scale**2)))
        x = x.reshape((B, -1, self.output_dim))
        x = self.norm(x)

        return x


class BasicUpLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default query and key scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
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
                 upsample=None):

        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=self.input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if
                                 (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution,
                                        dim=dim,
                                        dim_scale=2,
                                        norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = paddle.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
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

        huns = paddle.full_like(attn_mask, -100)
        attn_mask = huns * (attn_mask != 0).astype("float32")
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
