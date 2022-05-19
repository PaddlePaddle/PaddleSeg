"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from collections import OrderedDict

import paddle
import paddle.nn.functional as F

from paddle import nn

from .mynn import Norm2d, Upsample
from . import hrnetv2
from .para_init import constant_,kaiming_uniform_,kaiming_normal_
#from runx.logx import logx
#from config import cfg




def get_trunk(trunk_name = 'hrnetv2', output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, 'Only stride8 supported right now'

    
    if trunk_name == 'hrnetv2':
        backbone = hrnetv2.get_seg_model()
        high_level_ch = backbone.high_level_ch
        s2_ch = -1
        s4_ch = -1
    else:
        raise 'unknown backbone {}'.format(trunk_name)

    #logx.msg("Trunk: {}".format(trunk_name))
    return backbone, s2_ch, s4_ch, high_level_ch


class ConvBnRelu(nn.Layer):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=Norm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias_attr=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class AtrousSpatialPyramidPoolingModule(nn.Layer):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2D(in_dim, reduction_dim, kernel_size=1,
                                    bias_attr=False),
                          Norm2d(reduction_dim), nn.ReLU()))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2D(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias_attr=False),
                Norm2d(reduction_dim),
                nn.ReLU()
            ))
        self.features = nn.LayerList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2D(in_dim, reduction_dim, kernel_size=1, bias_attr=False),
            Norm2d(reduction_dim), nn.ReLU())

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = paddle.concat((out, y), 1)
        return out


class ASPP_edge(AtrousSpatialPyramidPoolingModule):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(ASPP_edge, self).__init__(in_dim=in_dim,
                                        reduction_dim=reduction_dim,
                                        output_stride=output_stride,
                                        rates=rates)
        self.edge_conv = nn.Sequential(
            nn.Conv2D(1, reduction_dim, kernel_size=1, bias_attr=False),
            Norm2d(reduction_dim), nn.ReLU())

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features
        edge_features = Upsample(edge, x_size[2:])
        edge_features = self.edge_conv(edge_features)
        out = paddle.concat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = paddle.concat((out, y), 1)
        return out


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2D(in_dim, reduction_dim, kernel_size=3, dilation=dil,
                  padding=dil, bias_attr=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU()
    )


class DPC(nn.Layer):
    '''
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = paddle.concat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch,
                                                 output_stride=output_stride)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


def BNReLU(ch):
    return nn.Sequential(
        Norm2d(ch),
        nn.ReLU())


def make_seg_head(in_ch, out_ch):
    #bot_ch = cfg.MODEL.SEGATTN_BOT_CH
    bot_ch = 256
    return nn.Sequential(
        nn.Conv2D(in_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False),
        Norm2d(bot_ch),
        nn.ReLU(),
        nn.Conv2D(bot_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False),
        Norm2d(bot_ch),
        nn.ReLU(),
        nn.Conv2D(bot_ch, out_ch, kernel_size=1, bias_attr=False))


def init_attn(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2D, nn.Linear)):
            constant_(module.weight,0)
            if module.bias is not None:
                constant_(module.bias, 0.5)
        #elif isinstance(module, cfg.MODEL.BNFUNC):
        elif isinstance(module, paddle.nn.BatchNorm2D):
            constant_(module.weight,0)
            constant_(module.bias,0)


def make_attn_head(in_ch, out_ch):
    #bot_ch = cfg.MODEL.SEGATTN_BOT_CH
    bot_ch = 256
    #if cfg.MODEL.MSCALE_OLDARCH:
    if False:
        return old_make_attn_head(in_ch, bot_ch, out_ch)

    od = OrderedDict([('conv0', nn.Conv2D(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias_attr=False)),
                      ('bn0', Norm2d(bot_ch)),
                      ('re0', nn.ReLU())])

    #if cfg.MODEL.MSCALE_INNER_3x3:
    if True:
        od['conv1'] = nn.Conv2D(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias_attr=False)
        od['bn1'] = Norm2d(bot_ch)
        od['re1'] = nn.ReLU()

    #if cfg.MODEL.MSCALE_DROPOUT:
    if False:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2D(bot_ch, out_ch, kernel_size=1, bias_attr=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential()
    for key in od:
        attn_head.add_sublayer(key,od[key])
        
    # init_attn(attn_head)
    return attn_head


def old_make_attn_head(in_ch, bot_ch, out_ch):
    attn = nn.Sequential(
        nn.Conv2D(in_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False),
        Norm2d(bot_ch),
        nn.ReLU(),
        nn.Conv2D(bot_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False),
        Norm2d(bot_ch),
        nn.ReLU(),
        nn.Conv2D(bot_ch, out_ch, kernel_size=out_ch, bias_attr=False),
        nn.Sigmoid())

    init_attn(attn)
    return attn
