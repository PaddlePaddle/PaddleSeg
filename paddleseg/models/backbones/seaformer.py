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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.models.backbones.transformer_utils import DropPath, zeros_, ones_
from paddleseg.utils import utils


class SeaFormer(nn.Layer):
    """
    The SeaFormer implementation based on PaddlePaddle.

    The original article refers to
    Qiang Wan, et, al. "SEAFORMER: SQUEEZE-ENHANCED AXIAL TRANSFORMER FOR MOBILE SEMANTIC SEGMENTATION"
    (https://arxiv.org/pdf/2301.13156.pdf).

    Args:
        in_channels (int, optional): The channels of input image. Default: 3.
        cfg (list[list], optional): Two values in the tuple indicate the indices of output of backbone. Defaulte: 
                      [[[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1]],
                       [[5, 3, 64, 2], [5, 3, 64, 1]],
                       [[3, 3, 128, 2], [3, 3, 128, 1]], 
                       [[5, 4, 192, 2]],
                       [[3, 6, 256, 2]]]
        channels (list[int], optional): Number of channels hidden layer. Default: [16, 32, 64, 128, 192, 256].
        embed_dims (list, optional): The size of embedding dimensions. Default: [192, 256].
        key_dims (list[int], optional): The unique number of key dimension. Default: [16, 24].
        depths (list[int], optional): The depth of every layer. Default: [4, 4].
        num_heads (int, optional): The number of heads for evary layer. Default: 8.
        attn_ratios (int, optional): Argument of expansion ratio of BasicLayer. Default: 2.
        mlp_ratios (list[int], optional): Argument of expansion ratio of MlP layers. Default: [2, 4].
        drop_path_ratio (int, optional): Argument of drop path ratio. Default: 0.1.
        act_layer (nn.Layer, optional): Activation function for model. Default: nn.ReLU6.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 cfgs=[[[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1]],
                       [[5, 3, 64, 2], [5, 3, 64, 1]],
                       [[3, 3, 128, 2], [3, 3, 128, 1]], [[5, 4, 192, 2]],
                       [[3, 6, 256, 2]]],
                 channels=[16, 32, 64, 128, 192, 256],
                 emb_dims=[192, 256],
                 key_dims=[16, 24],
                 depths=[4, 4],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.1,
                 act_layer=nn.ReLU6,
                 pretrained=None):
        super().__init__()

        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.pretrained = pretrained

        for i in range(len(cfgs)):
            smb = StackedMV2Block(
                in_channels=in_channels,
                cfgs=cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=channels[i])
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(depths)):
            dpr = [
                x.item() for x in paddle.linspace(0, drop_path_rate, depths[i])
            ]  # stochastic depth decay rule
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=emb_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0,
                attn_drop=0,
                drop_path=dpr,
                act_layer=act_layer)
            setattr(self, f"trans{i + 1}", trans)

        self.init_weights()

    def forward(self, x):
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            # 1/8 shared feat
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(
                    self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
                outputs.append(x)

        return outputs

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                std = layer._kernel_size[0] * layer._kernel_size[
                    1] * layer._out_channels
                std //= layer._groups
                param_init.normal_init(layer.weight, std=std)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
            elif isinstance(layer, nn.Linear):
                param_init.normal_init(layer.weight, std=0.01)
                if layer.bias is not None:
                    zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                zeros_(layer.bias)
                ones_(layer.weight)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.ConvBN(
            in_features, hidden_features, 1, bias_attr=False)
        self.dwconv = nn.Conv2D(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias_attr=True,
            groups=hidden_features)
        self.act = act_layer()
        self.fc2 = layers.ConvBN(
            hidden_features, out_features, 1, bias_attr=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Layer):
    def __init__(self,
                 inp: int,
                 oup: int,
                 ks: int,
                 stride: int,
                 expand_ratio: int,
                 activations=None):
        super().__init__()

        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        modules = []
        if expand_ratio != 1:
            modules.append(layers.ConvBN(inp, hidden_dim, 1, bias_attr=False))
            modules.append(activations())
        modules.extend([
            layers.ConvBN(
                hidden_dim,
                hidden_dim,
                ks,
                stride=stride,
                padding=ks // 2,
                groups=hidden_dim,
                bias_attr=False), activations(), layers.ConvBN(
                    hidden_dim, oup, 1, bias_attr=False)
        ])
        self.conv = nn.Sequential(*modules)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Layer):
    def __init__(self,
                 in_channels,
                 cfgs,
                 stem,
                 inp_channel=16,
                 activation=nn.ReLU,
                 width_mult=1.):
        super().__init__()

        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                layers.ConvBN(
                    in_channels,
                    inp_channel,
                    3,
                    stride=2,
                    padding=1,
                    bias_attr=False),
                activation())
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(
                inp_channel,
                output_channel,
                ks=k,
                stride=s,
                expand_ratio=t,
                activations=activation)
            self.add_sublayer(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for _, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Layer):
    def __init__(self, dim, shape):
        super().__init__()

        params = paddle.randn([1, dim, shape])
        self.pos_embed = self.create_parameter(
            shape=params.shape,
            dtype=str(params.numpy().dtype),
            default_initializer=nn.initializer.Assign(params))
        self.pos_embed.stop_gradient = False

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(
            self.pos_embed,
            size=[N],
            mode='linear',
            align_corners=False,
            data_format='NCW')
        return x


class SeaAttention(nn.Layer):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, activation=None):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = layers.ConvBN(dim, self.nh_kd, 1, bias_attr=False)
        self.to_k = layers.ConvBN(dim, self.nh_kd, 1, bias_attr=False)
        self.to_v = layers.ConvBN(dim, self.dh, 1, bias_attr=False)

        self.proj = nn.Sequential(
            activation(), layers.ConvBN(
                self.dh, dim, 1, bias_attr=False))
        self.proj_encode_row = nn.Sequential(
            activation(), layers.ConvBN(
                self.dh, self.dh, 1, bias_attr=False))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.proj_encode_column = nn.Sequential(
            activation(), layers.ConvBN(
                self.dh, self.dh, 1, bias_attr=False))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(self.nh_kd, 16)

        self.dwconv = layers.ConvBN(
            2 * self.dh,
            2 * self.dh,
            3,
            stride=1,
            padding=1,
            dilation=1,
            groups=2 * self.dh,
            bias_attr=False)
        self.act = activation()

        self.pwconv = layers.ConvBN(2 * self.dh, dim, 1, bias_attr=False)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhance
        qkv = paddle.concat([q, k, v], axis=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(
            [B, self.num_heads, -1, H]).transpose([0, 1, 3, 2])
        krow = self.pos_emb_rowk(k.mean(-1)).reshape([B, self.num_heads, -1, H])
        vrow = v.mean(-1).reshape([B, self.num_heads, -1, H]).transpose(
            [0, 1, 3, 2])

        attn_row = paddle.matmul(qrow, krow) * self.scale
        attn_row = F.softmax(attn_row, axis=-1)
        xx_row = paddle.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(
            xx_row.transpose([0, 1, 3, 2]).reshape([B, self.dh, H, 1]))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(
            [B, self.num_heads, -1, W]).transpose([0, 1, 3, 2])
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(
            [B, self.num_heads, -1, W])
        vcolumn = v.mean(-2).reshape([B, self.num_heads, -1, W]).transpose(
            [0, 1, 3, 2])

        attn_column = paddle.matmul(qcolumn, kcolumn) * self.scale
        attn_column = F.softmax(attn_column, axis=-1)
        xx_column = paddle.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(
            xx_column.transpose([0, 1, 3, 2]).reshape([B, self.dh, 1, W]))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = SeaAttention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Layer):
    def __init__(self,
                 block_num,
                 embedding_dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=None):
        super().__init__()

        self.block_num = block_num

        self.transformer_blocks = nn.LayerList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


@manager.BACKBONES.add_component
def SeaFormer_tiny(pretrained, **kwags):
    seaformer = SeaFormer(
        pretrained=pretrained,
        cfgs=[[[3, 1, 16, 1], [3, 4, 16, 2], [3, 3, 16, 1]],
              [[5, 3, 32, 2], [5, 3, 32, 1]], [[3, 3, 64, 2], [3, 3, 64, 1]],
              [[5, 3, 128, 2]], [[3, 6, 160, 2]]],
        emb_dims=[128, 160],
        channels=[16, 16, 32, 64, 128, 160],
        depths=[2, 2],
        num_heads=4,
        **kwags)
    seaformer.feat_channels = [32, 128, 160]

    return seaformer


@manager.BACKBONES.add_component
def SeaFormer_small(pretrained, **kwags):
    seaformer = SeaFormer(
        pretrained=pretrained,
        cfgs=[[[3, 1, 16, 1], [3, 4, 24, 2], [3, 3, 24, 1]],
              [[5, 3, 48, 2], [5, 3, 48, 1]], [[3, 3, 96, 2], [3, 3, 96, 1]],
              [[5, 4, 160, 2]], [[3, 6, 192, 2]]],
        emb_dims=[160, 192],
        channels=[16, 24, 48, 96, 160, 192],
        depths=[3, 3],
        num_heads=6,
        **kwags)
    seaformer.feat_channels = [48, 160, 192]

    return seaformer


@manager.BACKBONES.add_component
def SeaFormer_base(pretrained, **kwags):
    seaformer = SeaFormer(
        pretrained=pretrained,
        cfgs=[[[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1]],
              [[5, 3, 64, 2], [5, 3, 64, 1]], [[3, 3, 128, 2], [3, 3, 128, 1]],
              [[5, 4, 192, 2]], [[3, 6, 256, 2]]],
        emb_dims=[192, 256],
        key_dims=[16, 24],
        channels=[16, 32, 64, 128, 192, 256],
        depths=[4, 4],
        num_heads=8,
        mlp_ratios=[2, 4],
        **kwags)
    seaformer.feat_channels = [64, 192, 256]

    return seaformer


@manager.BACKBONES.add_component
def SeaFormer_large(pretrained, **kwags):
    seaformer = SeaFormer(
        pretrained=pretrained,
        cfgs=[[[3, 3, 32, 1], [3, 4, 64, 2], [3, 4, 64, 1]],
              [[5, 4, 128, 2], [5, 4, 128, 1]],
              [[3, 4, 192, 2], [3, 4, 192, 1]], [[5, 4, 256, 2]],
              [[3, 6, 320, 2]]],
        emb_dims=[192, 256, 320],
        key_dims=[16, 20, 24],
        channels=[32, 64, 128, 192, 256, 320],
        depths=[3, 3, 3],
        num_heads=8,
        mlp_ratios=[2, 4, 6],
        **kwags)
    seaformer.feat_channels = [128, 192, 256, 320]

    return seaformer
