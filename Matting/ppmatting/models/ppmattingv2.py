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

from functools import partial
from collections import defaultdict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddleseg
from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.models.backbones.transformer_utils import Identity, DropPath

from ppmatting.models.layers import MLFF
from ppmatting.models.losses import MRSD, GradientLoss


@manager.MODELS.add_component
class PPMattingV2(nn.Layer):
    """
    The PPMattingV2 implementation based on PaddlePaddle.

    The original article refers to
    TODO Guowei Chen, et, al. "" ().

    Args:
        backbone: backobne model.
        pretrained(str, optional): The path of pretrianed model. Defautl: None.
        dpp_len_trans(int, optional): The depth of transformer block in dpp(DoublePyramidPoolModule). Default: 1.
        dpp_index(list, optional): The index of backone output which as the input in dpp. Default: [1, 2, 3, 4].
        dpp_mid_channel(int, optional): The output channels of the first pyramid pool in dpp. Default: 256.
        dpp_out_channel(int, optional): The output channels of dpp. Default: 512.
        dpp_bin_sizes(list, optional): The output size of the second pyramid pool in dpp. Default: (2, 4, 6).
        dpp_mlp_ratios(int, optional): The expandsion ratio of mlp in dpp. Default: 2.
        dpp_attn_ratio(int, optional): The expandsion ratio of attention. Default: 2.
        dpp_merge_type(str, optional): The merge type of the output of the second pyramid pool in dpp, 
            which should be one of (`concat`, `add`). Default: 'concat'.
        mlff_merge_type(str, optional): The merge type of the multi features before output. 
            It should be one of ('add', 'concat'). Default: 'concat'.
    """

    def __init__(self,
                 backbone,
                 pretrained=None,
                 dpp_len_trans=1,
                 dpp_index=[1, 2, 3, 4],
                 dpp_mid_channel=256,
                 dpp_output_channel=512,
                 dpp_bin_sizes=(2, 4, 6),
                 dpp_mlp_ratios=2,
                 dpp_attn_ratio=2,
                 dpp_merge_type='concat',
                 mlff_merge_type='concat',
                 decoder_channels=[128, 96, 64, 32, 32],
                 head_channel=32):
        super().__init__()

        self.backbone = backbone
        self.backbone_channels = backbone.feat_channels

        # check
        assert len(backbone.feat_channels) == 5, \
            "Backbone should return 5 features with different scales"
        assert max(dpp_index) < len(backbone.feat_channels), \
            "The element of `dpp_index` should be less than the number of return features of backbone."

        # dpp module
        self.dpp_index = dpp_index
        self.dpp = DoublePyramidPoolModule(
            stride=2,
            input_channel=sum(self.backbone_channels[i]
                              for i in self.dpp_index),
            mid_channel=dpp_mid_channel,
            output_channel=dpp_output_channel,
            len_trans=dpp_len_trans,
            bin_sizes=dpp_bin_sizes,
            mlp_ratios=dpp_mlp_ratios,
            attn_ratio=dpp_attn_ratio,
            merge_type=dpp_merge_type)

        # decoder
        self.mlff32x = MLFF(
            in_channels=[self.backbone_channels[-1], dpp_output_channel],
            mid_channels=[dpp_output_channel, dpp_output_channel],
            out_channel=decoder_channels[0],
            merge_type=mlff_merge_type)
        self.mlff16x = MLFF(
            in_channels=[
                self.backbone_channels[-2], decoder_channels[0],
                dpp_output_channel
            ],
            mid_channels=[
                decoder_channels[0], decoder_channels[0], decoder_channels[0]
            ],
            out_channel=decoder_channels[1],
            merge_type=mlff_merge_type)
        self.mlff8x = MLFF(
            in_channels=[
                self.backbone_channels[-3], decoder_channels[1],
                dpp_output_channel
            ],
            mid_channels=[
                decoder_channels[1], decoder_channels[1], decoder_channels[1]
            ],
            out_channel=decoder_channels[2],
            merge_type=mlff_merge_type)
        self.mlff4x = MLFF(
            in_channels=[self.backbone_channels[-4], decoder_channels[2], 3],
            mid_channels=[decoder_channels[2], decoder_channels[2], 3],
            out_channel=decoder_channels[3])
        self.mlff2x = MLFF(
            in_channels=[self.backbone_channels[-5], decoder_channels[3], 3],
            mid_channels=[decoder_channels[3], decoder_channels[3], 3],
            out_channel=decoder_channels[4])

        self.matting_head_mlff8x = MattingHead(
            in_chan=decoder_channels[2], mid_chan=32)
        self.matting_head_mlff2x = MattingHead(
            in_chan=decoder_channels[4] + 3, mid_chan=head_channel, mid_num=2)

        # loss
        self.loss_func_dict = None

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        img = inputs['img']
        input_shape = paddle.shape(img)
        feats_backbone = self.backbone(
            img)  # stdc1 [2x, 4x, 8x, 16x, 32x] [32, 64, 256, 512, 1024]
        x = self.dpp([feats_backbone[i] for i in self.dpp_index])
        dpp_out = x

        input_32x = [feats_backbone[-1], x]
        x = self.mlff32x(input_32x,
                         paddle.shape(feats_backbone[-1])[-2:])  # 32x

        input_16x = [feats_backbone[-2], x, dpp_out]
        x = self.mlff16x(input_16x,
                         paddle.shape(feats_backbone[-2])[-2:])  # 16x

        input_8x = [feats_backbone[-3], x, dpp_out]
        x = self.mlff8x(input_8x, paddle.shape(feats_backbone[-3])[-2:])  # 8x
        mlff8x_output = x

        input_4x = [feats_backbone[-4], x]
        input_4x.append(
            F.interpolate(
                img, feats_backbone[-4].shape[2:], mode='area'))
        x = self.mlff4x(input_4x, paddle.shape(feats_backbone[-4])[-2:])  # 4x

        input_2x = [feats_backbone[-5], x]
        input_2x.append(
            F.interpolate(
                img, feats_backbone[-5].shape[2:], mode='area'))
        x = self.mlff2x(input_2x, paddle.shape(feats_backbone[-5])[-2:])  # 2x

        x = F.interpolate(
            x, input_shape[-2:], mode='bilinear', align_corners=False)
        x = paddle.concat([x, img], axis=1)
        alpha = self.matting_head_mlff2x(x)

        if self.training:
            logit_dict = {}
            logit_dict['alpha'] = alpha
            logit_dict['alpha_8x'] = self.matting_head_mlff8x(mlff8x_output)

            loss_dict = self.loss(logit_dict, inputs)

            return logit_dict, loss_dict
        else:
            return alpha

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            if self.loss_func_dict is None:
                self.loss_func_dict = defaultdict(list)
                self.loss_func_dict['alpha'].append(MRSD())
                self.loss_func_dict['alpha'].append(GradientLoss())
                self.loss_func_dict['alpha_8x'].append(MRSD())
                self.loss_func_dict['alpha_8x'].append(GradientLoss())
        else:
            self.loss_func_dict = loss_func_dict

        loss = {}
        alpha_8x_label = F.interpolate(
            label_dict['alpha'],
            size=logit_dict['alpha_8x'].shape[-2:],
            mode='area',
            align_corners=False)
        loss['alpha_8x_mrsd'] = self.loss_func_dict['alpha_8x'][0](
            logit_dict['alpha_8x'], alpha_8x_label)
        loss['alpha_8x_grad'] = self.loss_func_dict['alpha_8x'][1](
            logit_dict['alpha_8x'], alpha_8x_label)
        loss['alpha_8x'] = loss['alpha_8x_mrsd'] + loss['alpha_8x_grad']

        transition_mask = label_dict['trimap'] == 128
        loss['alpha_mrsd'] = self.loss_func_dict['alpha'][0](
            logit_dict['alpha'],
            label_dict['alpha']) + 2 * self.loss_func_dict['alpha'][0](
                logit_dict['alpha'], label_dict['alpha'], transition_mask)
        loss['alpha_grad'] = self.loss_func_dict['alpha'][1](
            logit_dict['alpha'],
            label_dict['alpha']) + 2 * self.loss_func_dict['alpha'][1](
                logit_dict['alpha'], label_dict['alpha'], transition_mask)
        loss['alpha'] = loss['alpha_mrsd'] + loss['alpha_grad']

        loss['all'] = loss['alpha'] + loss['alpha_8x']
        return loss

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class MattingHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, mid_num=1, out_channels=1):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.mid_conv = nn.LayerList([
            layers.ConvBNReLU(
                mid_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False) for i in range(mid_num - 1)
        ])
        self.conv_out = nn.Conv2D(
            mid_chan, out_channels, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        for mid_conv in self.mid_conv:
            x = mid_conv(x)
        x = self.conv_out(x)
        x = F.sigmoid(x)
        return x


class DoublePyramidPoolModule(nn.Layer):
    """
    Extract global information through double pyramid pool structure and attention calculation by transformer block.

    Args:
        stride(int): The stride for the inputs.
        input_channel(int): The total channels of input features.
        mid_channel(int, optional): The output channels of the first pyramid pool. Default: 256.
        out_channel(int, optional): The output channels. Default: 512.
        len_trans(int, optional): The depth of transformer block. Default: 1.
        bin_sizes(list, optional): The output size of the second pyramid pool. Default: (2, 4, 6).
        mlp_ratios(int, optional): The expandsion ratio of the mlp. Default: 2.
        attn_ratio(int, optional): The expandsion ratio of the attention. Default: 2.
        merge_type(str, optional): The merge type of the output of the second pyramid pool, which should be one of (`concat`, `add`). Default: 'concat'.
        align_corners(bool, optional): Whether to use `align_corners` when interpolating. Default: False.

    """

    def __init__(self,
                 stride,
                 input_channel,
                 mid_channel=256,
                 output_channel=512,
                 len_trans=1,
                 bin_sizes=(2, 4, 6),
                 mlp_ratios=2,
                 attn_ratio=2,
                 merge_type='concat',
                 align_corners=False):
        super().__init__()

        self.mid_channel = mid_channel
        self.align_corners = align_corners
        self.mlp_rations = mlp_ratios
        self.attn_ratio = attn_ratio
        if isinstance(len_trans, int):
            self.len_trans = [len_trans] * len(bin_sizes)
        elif isinstance(len_trans, (list, tuple)):
            self.len_trans = len_trans
            if len(len_trans) != len(bin_sizes):
                raise ValueError(
                    'If len_trans is list or tuple, the length should be same as bin_sizes'
                )
        else:
            raise ValueError(
                '`len_trans` only support int, list and tuple type')

        if merge_type not in ['add', 'concat']:
            raise ('`merge_type only support `add` or `concat`.')
        self.merge_type = merge_type

        self.pp1 = PyramidPoolAgg(stride=stride)
        self.conv_mid = layers.ConvBN(input_channel, mid_channel, 1)
        self.pp2 = nn.LayerList([
            self._make_stage(
                embdeding_channels=mid_channel, size=size, block_num=block_num)
            for size, block_num in zip(bin_sizes, self.len_trans)
        ])

        if self.merge_type == 'concat':
            in_chan = mid_channel + mid_channel * len(bin_sizes)
        else:
            in_chan = mid_channel
        self.conv_out = layers.ConvBNReLU(
            in_chan, output_channel, kernel_size=1)

    def _make_stage(self, embdeding_channels, size, block_num):
        prior = nn.AdaptiveAvgPool2D(output_size=size)
        if size == 1:
            trans = layers.ConvBNReLU(
                in_channels=embdeding_channels,
                out_channels=embdeding_channels,
                kernel_size=1)
        else:
            trans = BasicLayer(
                block_num=block_num,
                embedding_dim=embdeding_channels,
                key_dim=16,
                num_heads=8,
                mlp_ratios=self.mlp_rations,
                attn_ratio=self.attn_ratio,
                drop=0,
                attn_drop=0,
                drop_path=0,
                act_layer=nn.ReLU6,
                lr_mult=1.0)
        return nn.Sequential(prior, trans)

    def forward(self, inputs):
        x = self.pp1(inputs)
        pp2_input = self.conv_mid(x)

        cat_layers = []
        for stage in self.pp2:
            x = stage(pp2_input)
            x = F.interpolate(
                x,
                paddle.shape(pp2_input)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [pp2_input] + cat_layers[::-1]
        if self.merge_type == 'concat':
            cat = paddle.concat(cat_layers, axis=1)
        else:
            cat = sum(cat_layers)
        out = self.conv_out(cat)
        return out


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


class Attention(nn.Layer):
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
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_k = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_v = Conv2DBN(dim, self.dh, 1, lr_mult=lr_mult)

        self.proj = nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, dim, bn_weight_init=0, lr_mult=lr_mult))

    def forward(self, x):
        x_shape = paddle.shape(x)
        H, W = x_shape[2], x_shape[3]

        qq = self.to_q(x).reshape(
            [0, self.num_heads, self.key_dim, -1]).transpose([0, 1, 3, 2])
        kk = self.to_k(x).reshape([0, self.num_heads, self.key_dim, -1])
        vv = self.to_v(x).reshape([0, self.num_heads, self.d, -1]).transpose(
            [0, 1, 3, 2])

        attn = paddle.matmul(qq, kk)
        attn = F.softmax(attn, axis=-1)

        xx = paddle.matmul(attn, vv)

        xx = xx.transpose([0, 1, 3, 2]).reshape([0, self.dh, H, W])
        xx = self.proj(xx)
        return xx


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 mlp_ratios=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 lr_mult=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios

        self.attn = Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            lr_mult=lr_mult)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        mlp_hidden_dim = int(dim * mlp_ratios)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       lr_mult=lr_mult)

    def forward(self, x):
        h = x
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class BasicLayer(nn.Layer):
    def __init__(self,
                 block_num,
                 embedding_dim,
                 key_dim,
                 num_heads,
                 mlp_ratios=4.,
                 attn_ratio=2.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=None,
                 lr_mult=1.0):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.LayerList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratios=mlp_ratios,
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


class PyramidPoolAgg(nn.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.tmp = Identity()  # avoid the error of paddle.flops

    def forward(self, inputs):
        '''
        # The F.adaptive_avg_pool2d does not support the (H, W) be Tensor,
        # so exporting the inference model will raise error.
        _, _, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return paddle.concat(
            [F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], axis=1)
        '''
        out = []
        ks = 2**len(inputs)
        stride = self.stride**len(inputs)
        for x in inputs:
            x = F.avg_pool2d(x, int(ks), int(stride))
            ks /= 2
            stride /= 2
            out.append(x)
        out = paddle.concat(out, axis=1)
        return out
