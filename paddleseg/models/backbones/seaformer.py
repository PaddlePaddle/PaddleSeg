# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
"""
This file refers to https://github.com/hustvl/TopFormer and https://github.com/BR-IDL/PaddleViT
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg import utils
from paddleseg.models.backbones.transformer_utils import Identity, DropPath
from paddleseg.models.backbones.topformer_utils import *


def _make_divisible(val, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(val + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * val:
        new_v += divisor
    return new_v


class HSigmoid(nn.Layer):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class Conv2DBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 lr_mult=1.0):
        super().__init__()
        conv_weight_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.c = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            weight_attr=conv_weight_attr,
            bias_attr=False)
        bn_weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(bn_weight_init),
            learning_rate=lr_mult)
        bn_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0), learning_rate=lr_mult)
        self.bn = nn.BatchNorm2D(
            out_channels, weight_attr=bn_weight_attr, bias_attr=bn_bias_attr)

    def forward(self, inputs):
        out = self.c(inputs)
        out = self.bn(out)
        return out


class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.BatchNorm2D,
                 act=None,
                 bias_attr=False,
                 lr_mult=1.0,
                 use_conv=True):
        super(ConvBNAct, self).__init__()
        param_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=param_attr if bias_attr else False)
        self.act = act() if act is not None else Identity()
        self.bn = norm(out_channels, weight_attr=param_attr, bias_attr=param_attr) \
            if norm is not None else Identity()

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.,
                 lr_mult=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2DBN(in_features, hidden_features, lr_mult=lr_mult)
        param_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.dwconv = nn.Conv2D(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            groups=hidden_features,
            weight_attr=param_attr,
            bias_attr=param_attr)
        self.act = act_layer()
        self.fc2 = Conv2DBN(hidden_features, out_features, lr_mult=lr_mult)
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
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio,
                 activations=None,
                 lr_mult=1.0):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], "The stride should be 1 or 2."

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2DBN(
                    in_channels, hidden_dim, ks=1, lr_mult=lr_mult))
            layers.append(activations())
        layers.extend([
            Conv2DBN(
                hidden_dim,
                hidden_dim,
                ks=kernel_size,
                stride=stride,
                pad=kernel_size // 2,
                groups=hidden_dim,
                lr_mult=lr_mult), activations(), Conv2DBN(
                    hidden_dim, out_channels, ks=1, lr_mult=lr_mult)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Layer):
    def __init__(self,
                 cfgs,
                 stem,
                 inp_channel=16,
                 activation=nn.ReLU,
                 width_mult=1.,
                 lr_mult=1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2DBN(3, inp_channel, 3, 2, 1), activation())
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
                kernel_size=k,
                stride=s,
                expand_ratio=t,
                activations=activation,
                lr_mult=lr_mult)
            self.add_sublayer(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Layer):
    def __init__(self, dim, shape):
        super().__init__()
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal())
        self.pos_embed = paddle.create_parameter(
            shape=[1, dim, shape], attr=weight_attr, dtype='float32')

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(
            self.pos_embed,
            size=(N, ),
            mode='linear',
            align_corners=False,
            data_format='NCW')
        return x


class Sea_Attention(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 attn_ratio=4,
                 activation=None,
                 lr_mult=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_k = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_v = Conv2DBN(dim, self.dh, 1, lr_mult=lr_mult)

        self.proj = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, dim, bn_weight_init=0, lr_mult=lr_mult))
        self.proj_encode_row = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, self.dh, bn_weight_init=0, lr_mult=lr_mult))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, self.dh, bn_weight_init=0, lr_mult=lr_mult))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2DBN(
            2 * self.dh,
            2 * self.dh,
            ks=3,
            stride=1,
            pad=1,
            dilation=1,
            groups=2 * self.dh,
            lr_mult=lr_mult)
        self.act = activation()
        self.pwconv = Conv2DBN(2 * self.dh, dim, ks=1, lr_mult=lr_mult)
        self.sigmoid = HSigmoid()

    def forward(self, x):  # x (B,N,C)
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
        vcolumn = paddle.mean(v, -2).reshape(
            [B, self.num_heads, -1, W]).transpose([0, 1, 3, 2])

        attn_column = paddle.matmul(qcolumn, kcolumn) * self.scale
        attn_column = F.softmax(attn_column, axis=-1)
        xx_column = paddle.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(
            xx_column.transpose([0, 1, 3, 2]).reshape([B, self.dh, 1, W]))

        xx = paddle.add(xx_row, xx_column)
        xx = paddle.add(v, xx)
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
                 act_layer=nn.ReLU,
                 lr_mult=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Sea_Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            lr_mult=lr_mult)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       lr_mult=lr_mult)

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
                 lr_mult=1.0,
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
                    act_layer=act_layer,
                    lr_mult=lr_mult))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class Fusion_block(nn.Layer):
    def __init__(self, inp, oup, embed_dim, activations=None,
                 lr_mult=1.0) -> None:
        super(Fusion_block, self).__init__()
        self.local_embedding = ConvBNAct(
            inp, embed_dim, kernel_size=1, lr_mult=lr_mult)
        self.global_act = ConvBNAct(
            oup, embed_dim, kernel_size=1, lr_mult=lr_mult)
        self.act = HSigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape

        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act),
            size=(H, W),
            mode='bilinear',
            align_corners=False)
        out = local_feat * sig_act
        return out


class InjectionMultiSumallmultiallsum(nn.Layer):
    def __init__(self,
                 in_channels=(64, 128, 256, 384),
                 activations=None,
                 out_channels=256,
                 lr_mult=1.0):
        super(InjectionMultiSumallmultiallsum, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        self.out_channels = out_channels
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            self.act_list.append(HSigmoid())

    def forward(
            self, inputs
    ):  # x_x8, x_x16, x_x32, x_x64 3 outputs: [4, 64, 64, 64] [4, 192, 16, 16] [4, 256, 8, 8]
        low_feat1 = F.interpolate(
            inputs[0], scale_factor=0.5, mode="bilinear")  # x16
        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], scale_factor=2, mode="bilinear")  # x16
        low_feat2_act = self.act_list[1](
            self.act_embedding_list[1](low_feat2))  # x16
        low_feat2 = self.embedding_list[1](low_feat2)

        # low_feat3_act = F.interpolate(
        #     self.act_list[2](self.act_embedding_list[2](inputs[2])),
        #     size=low_feat2.shape[2:],
        #     mode="bilinear")
        # low_feat3 = F.interpolate(
        #     self.embedding_list[2](inputs[2]),
        #     size=low_feat2.shape[2:],
        #     mode="bilinear")

        high_feat_act = F.interpolate(
            self.act_list[2](self.act_embedding_list[2](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat1_act * low_feat2_act * high_feat_act * (
            low_feat1 + low_feat2) + high_feat

        return res


class SeaFormer(nn.Layer):
    def __init__(self,
                 cfgs,
                 channels,
                 embed_dims,
                 key_dims,
                 depths=[2, 2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.,
                 act_layer=nn.ReLU6,
                 lr_mult=1.0,
                 in_channels=3,
                 use_AAM=True,
                 pretrained=None):
        super().__init__()
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs

        for i in range(len(cfgs)):
            smb = StackedMV2Block(
                cfgs=cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=channels[i],
                lr_mult=lr_mult)
            setattr(self, f'smb{i+1}', smb)

        for i in range(len(depths)):
            dpr = [
                x.item() for x in paddle.linspace(0, drop_path_rate, depths[i])
            ]
            # dpr = [0 for _ in paddle.linspace(0, drop_path_rate, depths[i])]
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=embed_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0,
                attn_drop=0.0,
                drop_path=dpr,
                act_layer=act_layer,
                lr_mult=lr_mult)
            setattr(self, f"trans{i+1}", trans)

        self.use_AAM = use_AAM
        if self.use_AAM:
            self.inj_module = InjectionMultiSumallmultiallsum(
                in_channels=[channels[-4]] + channels[-2:],
                activations=act_layer,
                lr_mult=lr_mult)
            print('Using AAM')
            self.injection_out_channels = [self.inj_module.out_channels, ] * 3
        else:
            dims = [128, 160]
            channels = [64, 192, 256]
            for i in range(len(dims)):
                fuse = Fusion_block(
                    channels[0] if i == 0 else dims[i - 1],
                    channels[i + 1],
                    embed_dim=dims[i],
                    lr_mult=lr_mult)
                setattr(self, f"fuse{i + 1}", fuse)
            self.injection_out_channels = [dims[i]] * 3

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)

        # import numpy
        # numpy.random.seed(1234)
        # x = numpy.random.randn(4,3,512,512)
        # x = paddle.to_tensor(x, dtype='float32')

        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            # x.register_hook(lambda grad: print('x{} grad'.format(i), grad.abs().sum()))

            # 1/8 shared feat
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(
                    self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
                # x.register_hook(lambda grad: print('x_trans{} grad'.format(i), grad.abs().sum()))

                outputs.append(x)

        if self.use_AAM:
            output = self.inj_module(
                outputs
            )  # 3 outputs: [4, 64, 64, 64] [4, 192, 16, 16] [4, 256, 8, 8]
        else:
            x_detail = outputs[0]
            for i in range(2):
                fuse = getattr(self, f'fuse{i+1}')
                x_detail = fuse(x_detail, outputs[i + 1])
            output = x_detail

        return output


@manager.BACKBONES.add_component
def SeaFormer_Base(**kwargs):
    cfgs = [
        # k,  t,  c, s
        [3, 1, 16, 1],  # 1/2        
        [3, 4, 32, 2],  # 1/4 1      
        [3, 3, 32, 1],  #            
        [5, 3, 64, 2],  # 1/8 3      
        [5, 3, 64, 1],  #            
        [3, 3, 128, 2],  # 1/16 5     
        [3, 3, 128, 1],  #            
        [5, 6, 160, 2],  # 1/32 7     
        [5, 6, 160, 1],  #            
        [3, 6, 160, 1],  #            
    ]
    cfg1 = [
        # k,  t,  c, s
        [3, 1, 16, 1],
        [3, 4, 32, 2],
        [3, 3, 32, 1]
    ]
    cfg2 = [[5, 3, 64, 2], [5, 3, 64, 1]]
    cfg3 = [[3, 3, 128, 2], [3, 3, 128, 1]]
    cfg4 = [[5, 4, 192, 2]]
    cfg5 = [[3, 6, 256, 2]]
    channels = [16, 32, 64, 128, 192, 256]
    depths = [4, 4]
    key_dims = [16, 24]
    emb_dims = [192, 256]
    num_heads = 8
    drop_path_rate = 0.1

    model = SeaFormer(
        cfgs=[cfg1, cfg2, cfg3, cfg4, cfg5],
        channels=channels,
        embed_dims=emb_dims,
        key_dims=key_dims,
        depths=depths,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        act_layer=nn.ReLU6,
        # lr_mult=1.0,
        # in_channels=3,
        **kwargs)
    return model
