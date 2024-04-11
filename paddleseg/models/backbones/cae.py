# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from paddleseg.cvlibs import manager
from paddleseg.utils import utils, logger
from paddleseg.models.backbones.transformer_utils import to_2tuple, DropPath, Identity

zeros_ = Constant(value=0.)


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
                 proj_drop=0.,
                 window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=False)

        if qkv_bias:
            self.q_bias = self.create_parameter(shape=([dim]),
                                                default_initializer=zeros_)
            self.v_bias = self.create_parameter(shape=([dim]),
                                                default_initializer=zeros_)
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] -
                                          1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                shape=(self.num_relative_distance, num_heads),
                default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH

            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h,
                                                   coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            coords_flatten_1 = paddle.unsqueeze(coords_flatten, 2)
            coords_flatten_2 = paddle.unsqueeze(coords_flatten, 1)
            relative_coords = coords_flatten_1.clone() - coords_flatten_2.clone(
            )

            relative_coords = relative_coords.transpose(
                (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :,
                            0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                paddle.zeros(shape=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        x_shape = x.shape
        N, C = x_shape[1], x_shape[2]

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                (self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1
                ])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

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
                 window_size=None,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if init_values is not None:
            self.gamma_1 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size[1]

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size[0]

    @property
    def patch_shape(self):
        return (self.img_size[0] // self.patch_size[0],
                self.img_size[1] // self.patch_size[1])

    def forward(self, x):
        x = self.proj(x)
        return x


class RelativePositionBias(nn.Layer):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            shape=(self.num_relative_distance, num_heads),
            default_initialize=zeros_)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h,
                                               coords_w]))  # 2, Wh, Ww
        coords_flatten = coords.flatten(1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpos((1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:,
                                1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                 self.window_size[0] * self.window_size[1] + 1,
                 self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose((2, 0, 1))  # nH, Wh*Ww, Wh*Ww


def get_sinusoid_encoding_table(n_position, d_hid, token=False):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if token:
        sinusoid_table = np.concatenate(
            [sinusoid_table, np.zeros([1, d_hid])], dim=0)
    return Tensor(sinusoid_table, dtype=float32).unsqueeze(0)


@manager.BACKBONES.add_component
class CAE(nn.Layer):
    """
    The Context Autoencoder for Self-Supervised Representation Learning implemetation based on PaddlePaddle

    The original article refers to Chen, Xiaokang, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. "Context autoencoder for self-supervised representation learning." arXiv preprint arXiv:2202.03026 (2022).
    (https://arxiv.org/abs/2202.03026)

    Args:
        img_size (int): Input image size for training the pretrained model, used in absolute postion embedding. Default: 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        init_values(float): The initial value of dropout in the block. Default: None.
        use_rel_pos_bias(bool): Whether to use relative position bias. Default: False.
        use_shared_rel_pos_bias(bool): Whether to use relative position bias. Default: False.
        epsilon(float): Epsilon in first norm of block. Default: 1e-5.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
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
                 init_values=None,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 epsilon=1e-5,
                 pretrained=None,
                 **args):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size,
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

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.use_rel_pos_bias = use_rel_pos_bias

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  init_values=init_values,
                  window_size=self.patch_embed.patch_shape
                  if use_rel_pos_bias else None,
                  epsilon=epsilon) for i in range(depth)
        ])

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained:
            utils.load_pretrained_model(self, self.pretrained)

        model_path = self.pretrained
        if not os.path.exists(model_path):
            model_path = utils.download_pretrained_model(model_path)

        load_state_dict = paddle.load(model_path)
        model_state_dict = self.state_dict()
        pos_embed_name = "pos_embed"
        if pos_embed_name in load_state_dict.keys():
            load_pos_embed = paddle.to_tensor(load_state_dict[pos_embed_name],
                                              dtype="float32")
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
        pos_embed = F.interpolate(pos_embed,
                                  new_hw,
                                  mode='bicubic',
                                  align_corners=False)
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        pos_embed = paddle.concat([cls_pos_embed, pos_embed], axis=1)

        return pos_embed

    def forward(self, x):
        x = self.patch_embed(x)
        x_shape = x.shape  # b * c * h * w

        cls_tokens = self.cls_token.expand((x_shape[0], -1, -1))
        x = x.flatten(2).transpose([0, 2, 1])  # b * hw * c
        x = paddle.concat([cls_tokens, x], axis=1)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            x = x + self.resize_pos_embed(self.pos_embed,
                                          (self.pos_h, self.pos_w), x_shape[2:])
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None

        res = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias)
            res.append(x[:, 1:, :])
        return res, x_shape

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


@manager.BACKBONES.add_component
def CAE_small_patch16_224(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=8,
                              num_heads=8,
                              mlp_ratio=3,
                              qk_scale=768**-0.5,
                              **kwargs)
    return model


@manager.BACKBONES.add_component
def CAE_base_patch16_224(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              epsilon=1e-6,
                              **kwargs)
    return model


@manager.BACKBONES.add_component
def CAE_base_patch16_384(**kwargs):
    model = VisionTransformer(img_size=384,
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
def CAE_base_patch32_384(**kwargs):
    model = VisionTransformer(img_size=384,
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
def CAE_large_patch16_224(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              epsilon=1e-6,
                              **kwargs)
    return model


@manager.BACKBONES.add_component
def CAE_large_patch16_384(**kwargs):
    model = VisionTransformer(img_size=384,
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
def CAE_large_patch32_384(**kwargs):
    model = VisionTransformer(img_size=384,
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
def CAE_huge_patch16_224(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              mlp_ratio=4,
                              **kwargs)
    return model


@manager.BACKBONES.add_component
def CAE_huge_patch32_384(**kwargs):
    model = VisionTransformer(img_size=384,
                              patch_size=32,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              mlp_ratio=4,
                              **kwargs)
    return model
