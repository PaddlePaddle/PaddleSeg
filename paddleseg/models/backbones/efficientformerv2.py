import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

import itertools
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.backbones.transformer_utils import *

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(nn.Layer):
    def __init__(self,
                 dim=384,
                 key_dim=32,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2D(
                    dim,
                    dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=dim),
                nn.BatchNorm2D(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution**2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(
            nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(
            nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(
            nn.Conv2D(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2D(self.num_heads * self.d), )
        self.v_local = nn.Sequential(
            nn.Conv2D(
                self.num_heads * self.d,
                self.num_heads * self.d,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.num_heads * self.d),
            nn.BatchNorm2D(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2D(
            self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2D(
            self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2D(self.dh, dim, 1),
            nn.BatchNorm2D(dim), )

        points = list(
            itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', paddle.zeros([num_heads, 49]))

        self.register_buffer(
            'attention_bias_idxs', paddle.ones(
                [49, 49], dtype=paddle.int64))

        self.attention_biases_seg = self.create_parameter(
            shape=[num_heads, len(attention_offsets)],
            default_initializer=zeros_)
        self.register_buffer('attention_bias_idxs_seg',
                             paddle.to_tensor(idxs).reshape([N, N]))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H = H // 2
            W = W // 2

        q = self.q(x).flatten(2).reshape(
            [B, self.num_heads, -1, H * W]).transpose([0, 1, 3, 2])
        k = self.k(x).flatten(2).reshape(
            [B, self.num_heads, -1, H * W]).transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose(
            [0, 1, 3, 2])

        attn = (q @k) * self.scale
        bias = paddle.gather(
            self.attention_biases_seg,
            self.attention_bias_idxs_seg.flatten(),
            axis=1).reshape([
                self.attention_biases_seg.shape[0],
                self.attention_bias_idxs_seg.shape[0],
                self.attention_bias_idxs_seg.shape[1]
            ])  # if self.training else self.ab

        bias = F.interpolate(
            bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = (attn @v)

        out = x.transpose([0, 1, 3, 2]).reshape([B, self.dh, H, W]) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2D(
            in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs // 2),
        act_layer(),
        nn.Conv2D(
            out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs),
        act_layer(), )


class LGQuery(nn.Layer):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2D(1, 2, 0)
        self.local = nn.Sequential(
            nn.Conv2D(
                in_dim,
                in_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_dim), )
        self.proj = nn.Sequential(
            nn.Conv2D(in_dim, out_dim, 1),
            nn.BatchNorm2D(out_dim), )

    def forward(self, x):
        B, C, H, W = x.shape
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(nn.Layer):
    def __init__(
            self,
            dim=384,
            key_dim=16,
            num_heads=8,
            attn_ratio=4,
            resolution=7,
            out_dim=None,
            act_layer=None, ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim

        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution,
                         self.resolution2)

        self.N = self.resolution**2
        self.N2 = self.resolution2**2

        self.k = nn.Sequential(
            nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(
            nn.Conv2D(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2D(self.num_heads * self.d), )
        self.v_local = nn.Sequential(
            nn.Conv2D(
                self.num_heads * self.d,
                self.num_heads * self.d,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.num_heads * self.d),
            nn.BatchNorm2D(self.num_heads * self.d), )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2D(self.dh, self.out_dim, 1),
            nn.BatchNorm2D(self.out_dim), )

        points = list(
            itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(
            itertools.product(
                range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) -
                        p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) -
                        p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', paddle.zeros([num_heads, 196]))
        self.register_buffer(
            'attention_bias_idxs', paddle.ones(
                [49, 196], dtype=paddle.int64))

        self.attention_biases_seg = self.create_parameter(
            shape=[num_heads, len(attention_offsets)],
            default_initializer=zeros_)

        self.register_buffer(
            'attention_bias_idxs_seg',
            paddle.to_tensor(
                idxs, dtype=paddle.int64).reshape([N_, N]))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape(
            [B, self.num_heads, -1, H * W // 4]).transpose([0, 1, 3, 2])
        k = self.k(x).flatten(2).reshape(
            [B, self.num_heads, -1, H * W]).transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose(
            [0, 1, 3, 2])

        attn = (q @k) * self.scale
        bias = paddle.gather(
            self.attention_biases_seg,
            self.attention_bias_idxs_seg.flatten(),
            axis=1).reshape([
                self.attention_biases_seg.shape[0],
                self.attention_bias_idxs_seg.shape[0],
                self.attention_bias_idxs_seg.shape[1]
            ])

        bias = F.interpolate(
            bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = F.softmax(attn, axis=-1)

        x = (attn @v).transpose([0, 1, 3, 2])
        out = x.reshape([B, self.dh, H // 2, W // 2]) + v_local

        out = self.proj(out)
        return out


class Embedding(nn.Layer):
    def __init__(self,
                 patch_size=3,
                 stride=2,
                 padding=1,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=nn.BatchNorm2D,
                 light=False,
                 asub=False,
                 resolution=None,
                 act_layer=nn.ReLU,
                 attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2D(
                    in_chans,
                    in_chans,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_chans),
                nn.BatchNorm2D(in_chans),
                nn.Hardswish(),
                nn.Conv2D(
                    in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(embed_dim), )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2D(embed_dim))
        elif self.asub:
            self.attn = attn_block(
                dim=in_chans,
                out_dim=embed_dim,
                resolution=resolution,
                act_layer=act_layer)

            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)

            out = self.attn(x) + out_conv

        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        # self.mid_conv = False
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2D(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_features)

        self.norm1 = nn.BatchNorm2D(hidden_features)
        self.norm2 = nn.BatchNorm2D(out_features)
        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.kaiming_normal_init(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class AttnFFN(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 resolution=7,
                 stride=None):

        super().__init__()

        self.token_mixer = Attention4D(
            dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            param_value = layer_scale_init_value * paddle.ones(
                [dim]).unsqueeze(-1).unsqueeze(-1)
            self.layer_scale_1 = self.create_parameter(
                shape=param_value.shape,
                dtype=str(param_value.numpy().dtype),
                default_initializer=nn.initializer.Assign(param_value))

            self.layer_scale_2 = self.create_parameter(
                shape=param_value.shape,
                dtype=str(param_value.numpy().dtype),
                default_initializer=nn.initializer.Assign(param_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Layer):
    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            param_value_2 = layer_scale_init_value * paddle.ones(
                [dim]).unsqueeze(-1).unsqueeze(-1)
            self.layer_scale_2 = self.create_parameter(
                shape=param_value_2.shape,
                dtype=str(param_value_2.numpy().dtype),
                default_initializer=nn.initializer.Assign(param_value_2))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim,
                index,
                layers,
                pool_size=3,
                mlp_ratio=4.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                drop_rate=.0,
                drop_path_rate=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
                vit_num=1,
                resolution=7,
                e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(
                AttnFFN(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    resolution=resolution,
                    stride=stride, ))
        else:
            blocks.append(
                FFN(
                    dim,
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value, ))

    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(nn.Layer):
    def __init__(self,
                 layers,
                 in_channels=3,
                 mode='multi_scale',
                 embed_dims=None,
                 mlp_ratios=4,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2D,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 fork_feat=False,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 resolution=512,
                 e_ratios=expansion_ratios_L,
                 **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.mode = mode

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.feat_channels = [32, 64, 144, 288]

        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=math.ceil(resolution / (2**(i + 2))),
                vit_num=vit_num,
                e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2**(i + 2))),
                        asub=asub,
                        act_layer=act_layer,
                        norm_layer=norm_layer, ))

        self.network = nn.LayerList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        if self.fork_feat:
            self.init_weight()
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
            elif isinstance(layer, nn.Linear):
                trunc_normal_(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                zeros_(layer.bias)
                ones_(layer.weight)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                # norm_layer = getattr(self, f'norm{idx}')
                # x_out = norm_layer(x)
                outs.append(x)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            if self.mode is not 'multi_scale':
                x = paddle.concat(
                    [
                        F.interpolate(
                            feat, size=x[0].shape[-2:], mode='bilinear')
                        for feat in x
                    ],
                    axis=1)
                # otuput features of four stages for dense prediction
                self.feat_channels = [sum(self.feat_channels)]
                return [x]
            else:
                return x
        # print(x.size())
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(
                x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out


@manager.BACKBONES.add_component
class efficientformerv2_s0_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S0'],
            embed_dims=EfficientFormer_width['S0'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.,
            vit_num=2,
            e_ratios=expansion_ratios_S0,
            **kwargs)


@manager.BACKBONES.add_component
class efficientformerv2_s1_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S1'],
            embed_dims=EfficientFormer_width['S1'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.,
            vit_num=2,
            e_ratios=expansion_ratios_S1,
            **kwargs)


@manager.BACKBONES.add_component
class efficientformerv2_s2_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S2'],
            embed_dims=EfficientFormer_width['S2'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.02,
            vit_num=4,
            e_ratios=expansion_ratios_S2,
            **kwargs)


@manager.BACKBONES.add_component
class efficientformerv2_l_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['L'],
            embed_dims=EfficientFormer_width['L'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.1,
            vit_num=6,
            e_ratios=expansion_ratios_L,
            **kwargs)
