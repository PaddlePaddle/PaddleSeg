# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.utils import utils, logger
from paddleseg.models.backbones.transformer_utils import to_2tuple, DropPath, Identity


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


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_shape = paddle.shape(x)
        N, C = x_shape[1], x_shape[2]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads,
                                   C // self.num_heads)).transpose((2, 0, 3, 1,
                                                                    4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size[1]

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size[0]

    def forward(self, x):
        x = self.proj(x)
        return x


@manager.BACKBONES.add_component
class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 final_norm=False,
                 pretrained=None,
                 **args):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        self.pos_w = self.patch_embed.num_patches_in_w
        self.pos_h = self.patch_embed.num_patches_in_h

        self.pos_embed = self.create_parameter(
            shape=(1, self.pos_w * self.pos_h + 1, embed_dim),
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.final_norm = final_norm
        if self.final_norm:
            self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)

        # load and resize pos_embed
        model_path = self.pretrained
        if not os.path.exists(model_path):
            model_path = utils.download_pretrained_model(model_path)

        load_state_dict = paddle.load(model_path)
        model_state_dict = self.state_dict()
        pos_embed_name = "pos_embed"
        if pos_embed_name in load_state_dict.keys():
            load_pos_embed = paddle.to_tensor(
                load_state_dict[pos_embed_name], dtype="float32")
            if self.pos_embed.shape != load_pos_embed.shape:
                pos_size = int(math.sqrt(load_pos_embed.shape[1] - 1))
                model_state_dict[pos_embed_name] = self.resize_pos_embed(
                    load_pos_embed, (pos_size, pos_size),
                    (self.pos_h, self.pos_w))
                self.set_dict(model_state_dict)
                logger.info(
                    "Load pos_embed and resize it from {} to {} .".format(
                        load_pos_embed.shape, self.pos_embed.shape))

    def resize_pos_embed(self, pos_embed, old_hw, new_hw):
        """
        Resize pos_embed weight.
        Args:
            pos_embed (Tensor): the pos_embed weight
            old_hw (list[int]): the height and width of old pos_embed
            new_hw (list[int]): the height and width of new pos_embed
        Returns:
            Tensor: the resized pos_embed weight
        """
        cls_pos_embed = pos_embed[:, :1, :]
        pos_embed = pos_embed[:, 1:, :]

        pos_embed = pos_embed.transpose([0, 2, 1])
        pos_embed = pos_embed.reshape([1, -1, old_hw[0], old_hw[1]])
        pos_embed = F.interpolate(
            pos_embed, new_hw, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        pos_embed = paddle.concat([cls_pos_embed, pos_embed], axis=1)

        return pos_embed

    def forward(self, x):
        x = self.patch_embed(x)
        x_shape = paddle.shape(x)  # b * c * h * w

        cls_tokens = self.cls_token.expand((x_shape[0], -1, -1))
        x = x.flatten(2).transpose([0, 2, 1])  # b * hw * c
        x = paddle.concat([cls_tokens, x], axis=1)

        if paddle.shape(x)[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            x = x + self.resize_pos_embed(self.pos_embed,
                                          (self.pos_h, self.pos_w), x_shape[2:])
        x = self.pos_drop(x)

        res = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.final_norm and idx == len(self.blocks) - 1:
                x = self.norm(x)
            res.append(x[:, 1:, :])

        return res, x_shape


@manager.BACKBONES.add_component
def ViT_small_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_base_patch16_384(**kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_base_patch32_384(**kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_large_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_large_patch16_384(**kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_large_patch32_384(**kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_huge_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def ViT_huge_patch32_384(**kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        **kwargs)
    return model
