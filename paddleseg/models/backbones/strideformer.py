# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
This file refers to https://github.com/hustvl/TopFormer and https://github.com/fudan-zvg/SeaFormer
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import regularizer

from paddleseg.cvlibs import manager
from paddleseg import utils
from paddleseg.models.backbones.transformer_utils import DropPath
from paddleseg.models.backbones.mobilenetv3 import _make_divisible, _create_act, Hardsigmoid
from paddleseg.models.layers import layer_libs


class StrideFormer(nn.Layer):
    def __init__(self,
                 cfgs,
                 channels,
                 embed_dims,
                 key_dims=[16, 24],
                 depths=[2, 2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.1,
                 act_layer=nn.Sigmoid,
                 inj_type='AAM',
                 out_channels=256,
                 dims=(128, 160),
                 out_feat_chs=None,
                 stride_attention=True,
                 in_channels=3,
                 pretrained=None):
        """
        The StrideFormer implementation based on PaddlePaddle.

        The original article refers to:

        Args:
            cfgs(list): Each sublist describe the config for a MobileNetV3 block.
            channels(list): The input channels for each MobileNetV3 block.
            embed_dims(list): The channels of the features input to the sea attention block.  
            key_dims(list, optional): The embeding dims for each head in attention.
            depths(list, optional): describes the depth of the attention block. i,e: M,N.
            num_heads(int, optional): The number of heads of the attention blocks.
            attn_ratios(int, optional): The exapend ratio of V.
            mlp_ratios(list, optional): The ratio of mlp blocks.
            drop_path_rate(float, optional): The drop path rate in attention block.
            act_layer(nn.Layer, optional): The activation layer of AAM.
            inj_type(string, optional): The type of injection/AAM.
            out_channels(int, optional): The output channels of the AAM.
            dims(list, optional): The dimension of the fusion block.
            out_feat_chs(list, optional): The input channels of the AAM.
            stride_attention(bool, optioal): whether to stride attention in each attention layer.
            in_channels (int, optional): The channels of input image. Default: 3.
            pretrained(str, optional): the path of pretrained model.
        """
        super().__init__()
        self.depths = depths
        self.cfgs = cfgs
        self.dims = dims

        for i in range(len(cfgs)):
            smb = StackedMV3Block(
                cfgs=cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=channels[i])
            setattr(self, f'smb{i+1}', smb)

        for i in range(len(depths)):
            dpr = [
                x.item() for x in paddle.linspace(0, drop_path_rate, depths[i])
            ]
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
                stride_attention=stride_attention)
            setattr(self, f"trans{i+1}", trans)

        self.inj_type = inj_type
        if self.inj_type == "AAM":
            self.inj_module = InjectionMultiSumallmultiallsum(
                in_channels=out_feat_chs, out_channels=out_channels)
            self.feat_channels = [out_channels, ]
        elif self.inj_type == "AAMSx8":
            self.inj_module = InjectionMultiSumallmultiallsumSimpx8(
                in_channels=out_feat_chs, out_channels=out_channels)
            self.feat_channels = [out_channels, ]
        elif self.inj_type == 'origin':
            for i in range(len(dims)):
                fuse = Fusion_block(
                    out_feat_chs[0] if i == 0 else dims[i - 1],
                    out_feat_chs[i + 1],
                    embed_dim=dims[i])
                setattr(self, f"fuse{i + 1}", fuse)
            self.feat_channels = [dims[i], ]
        else:
            raise NotImplementedError(self.inj_module + ' is not implemented')

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

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

        if self.inj_type == "origin":
            x_detail = outputs[0]
            for i in range(len(self.dims)):
                fuse = getattr(self, f'fuse{i+1}')

                x_detail = fuse(x_detail, outputs[i + 1])
            output = x_detail
        else:
            output = self.inj_module(outputs)

        return output


class HSigmoid(nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


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
                 bias_attr=False):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=None if bias_attr else False)
        self.act = act() if act is not None else nn.Identity()
        self.bn = norm(out_channels, bias_attr=None) \
            if norm is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv2DBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1):
        super().__init__()
        self.c = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias_attr=False)
        bn_weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(bn_weight_init))
        bn_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        self.bn = nn.BatchNorm2D(
            out_channels, weight_attr=bn_weight_attr, bias_attr=bn_bias_attr)

    def forward(self, inputs):
        out = self.c(inputs)
        out = self.bn(out)
        return out


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 dilation=1):
        super().__init__()

        self.c = nn.Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False,
            dilation=dilation)
        self.bn = nn.BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=paddle.ParamAttr(regularizer=regularizer.L2Decay(0.0)),
            bias_attr=paddle.ParamAttr(regularizer=regularizer.L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 dilation=1):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2) * dilation,
            num_groups=mid_c,
            if_act=True,
            act=act,
            dilation=dilation)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


class StackedMV3Block(nn.Layer):
    """
    The MobileNetV3 block.

    Args:
        cfgs (list): The MobileNetV3 config list of a stage.
        stem (bool): Whether is the first stage or not.
        in_channels (int, optional): The channels of input image. Default: 3.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
    
    Returns:
        model: nn.Layer. A stage of specific MobileNetV3 model depends on args.
    """

    def __init__(self, cfgs, stem, inp_channel, scale=1.0):
        super().__init__()

        self.scale = scale
        self.stem = stem

        if self.stem:
            self.conv = ConvBNLayer(
                in_c=3,
                out_c=_make_divisible(inp_channel * self.scale),
                filter_size=3,
                stride=2,
                padding=1,
                num_groups=1,
                if_act=True,
                act="hardswish")
        self.blocks = nn.LayerList()
        for i, (k, exp, c, se, act, s) in enumerate(cfgs):
            self.blocks.append(
                ResidualUnit(
                    in_c=_make_divisible(inp_channel * self.scale),
                    mid_c=_make_divisible(self.scale * exp),
                    out_c=_make_divisible(self.scale * c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=act,
                    dilation=1))
            inp_channel = _make_divisible(self.scale * c)

    def forward(self, x):
        if self.stem:
            x = self.conv(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

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
                 stride_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2DBN(dim, nh_kd, 1)
        self.to_k = Conv2DBN(dim, nh_kd, 1)
        self.to_v = Conv2DBN(dim, self.dh, 1)

        self.stride_attention = stride_attention

        if self.stride_attention:
            self.stride_conv = nn.Sequential(
                nn.Conv2D(
                    dim, dim, kernel_size=3, stride=2, padding=1, groups=dim),
                nn.BatchNorm2D(dim), )

        self.proj = paddle.nn.Sequential(
            activation(), Conv2DBN(
                self.dh, dim, bn_weight_init=0))
        self.proj_encode_row = paddle.nn.Sequential(
            activation(), Conv2DBN(
                self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = paddle.nn.Sequential(
            activation(), Conv2DBN(
                self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2DBN(
            2 * self.dh,
            2 * self.dh,
            ks=3,
            stride=1,
            pad=1,
            dilation=1,
            groups=2 * self.dh)
        self.act = activation()
        self.pwconv = Conv2DBN(2 * self.dh, dim, ks=1)
        self.sigmoid = HSigmoid()

    def forward(self, x):
        B, C, H_ori, W_ori = x.shape
        if self.stride_attention:
            x = self.stride_conv(x)
        B, C, H, W = x.shape

        q = self.to_q(x)  # [B, nhead*dim, H, W]
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = paddle.concat([q, k, v], axis=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(
            [B, self.num_heads, -1, H]).transpose(
                [0, 1, 3, 2])  # [B, nhead, H, dim] 
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(
            [B, self.num_heads, -1, H])  # [B, nhead, dim, H]
        vrow = v.mean(-1).reshape([B, self.num_heads, -1, H]).transpose(
            [0, 1, 3, 2])  # [B, nhead, H, dim*attn_ratio]

        attn_row = paddle.matmul(qrow, krow) * self.scale  # [B, nhead, H, H]
        attn_row = F.softmax(attn_row, axis=-1)

        xx_row = paddle.matmul(attn_row, vrow)  # [B, nhead, H, dim*attn_ratio]
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

        xx = paddle.add(xx_row, xx_column)  # [B, self.dh, H, W] 
        xx = paddle.add(v, xx)

        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        if self.stride_attention:
            xx = F.interpolate(xx, size=(H_ori, W_ori), mode='bilinear')

        return xx


class MLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2DBN(in_features, hidden_features)
        self.dwconv = nn.Conv2D(
            hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2DBN(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
                 stride_attention=None):
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
            stride_attention=stride_attention)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
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
                 act_layer=None,
                 stride_attention=None):
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
                    stride_attention=stride_attention))

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class Fusion_block(nn.Layer):
    def __init__(self, inp, oup, embed_dim) -> None:
        super(Fusion_block, self).__init__()
        self.local_embedding = ConvBNAct(inp, embed_dim, kernel_size=1)
        self.global_act = ConvBNAct(oup, embed_dim, kernel_size=1)
        self.act = nn.Sigmoid()

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
    def __init__(self, in_channels=(64, 128, 256, 384), out_channels=256):
        super(InjectionMultiSumallmultiallsum, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_list.append(nn.Sigmoid())

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64 
        low_feat1 = F.interpolate(inputs[0], scale_factor=0.5, mode="bilinear")
        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")
        low_feat2_act = self.act_list[1](
            self.act_embedding_list[1](low_feat2))  # x16
        low_feat2 = self.embedding_list[1](low_feat2)

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


class InjectionMultiSumallmultiallsumSimpx8(nn.Layer):
    def __init__(self, in_channels=(64, 128, 256, 384), out_channels=256):
        super(InjectionMultiSumallmultiallsumSimpx8, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        for i in range(len(in_channels)):
            if i != 1:
                self.embedding_list.append(
                    ConvBNAct(
                        in_channels[i], out_channels, kernel_size=1))
            if i != 0:
                self.act_embedding_list.append(
                    ConvBNAct(
                        in_channels[i], out_channels, kernel_size=1))
                self.act_list.append(nn.Sigmoid())

    def forward(self, inputs):
        # x_x8, x_x16, x_x32
        low_feat1 = self.embedding_list[0](inputs[0])

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")
        low_feat2_act = self.act_list[0](self.act_embedding_list[0](low_feat2))

        high_feat_act = F.interpolate(
            self.act_list[1](self.act_embedding_list[1](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[1](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat2_act * high_feat_act * low_feat1 + high_feat

        return res


@manager.BACKBONES.add_component
def MobileSeg_Base(**kwargs):
    cfg1 = [
        # k t c, s
        [3, 16, 16, True, "relu", 1],
        [3, 64, 32, False, "relu", 2],
        [3, 96, 32, False, "relu", 1]
    ]
    cfg2 = [[5, 128, 64, True, "hardswish", 2],
            [5, 240, 64, True, "hardswish", 1]]
    cfg3 = [[5, 384, 128, True, "hardswish", 2],
            [5, 384, 128, True, "hardswish", 1]]
    cfg4 = [[5, 768, 192, True, "hardswish", 2],
            [5, 768, 192, True, "hardswish", 1]]

    channels = [16, 32, 64, 128, 192]
    depths = [3, 3]
    emb_dims = [128, 192]
    num_heads = 8

    model = StrideFormer(
        cfgs=[cfg1, cfg2, cfg3, cfg4],
        channels=channels,
        embed_dims=emb_dims,
        depths=depths,
        num_heads=num_heads,
        act_layer=nn.ReLU6,
        **kwargs)

    return model


@manager.BACKBONES.add_component
def MobileSeg_Tiny(**kwargs):
    cfg1 = [
        # k t c, s
        [3, 16, 16, True, "relu", 1],
        [3, 64, 32, False, "relu", 2],
        [3, 48, 24, False, "relu", 1]
    ]
    cfg2 = [[5, 96, 32, True, "hardswish", 2],
            [5, 96, 32, True, "hardswish", 1]]
    cfg3 = [[5, 160, 64, True, "hardswish", 2],
            [5, 160, 64, True, "hardswish", 1]]
    cfg4 = [[3, 384, 128, True, "hardswish", 2],
            [3, 384, 128, True, "hardswish", 1]]

    channels = [16, 24, 32, 64, 128]
    depths = [2, 2]
    emb_dims = [64, 128]
    num_heads = 4

    model = StrideFormer(
        cfgs=[cfg1, cfg2, cfg3, cfg4],
        channels=channels,
        embed_dims=emb_dims,
        depths=depths,
        num_heads=num_heads,
        act_layer=nn.ReLU6,
        **kwargs)

    return model
