# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.backbones.transformer_utils import (DropPath, Identity)
from paddleseg.cvlibs.param_init import (constant_init, kaiming_normal_init,
                                         trunc_normal_init)


@manager.MODELS.add_component
class RTFormer(nn.Layer):
    """
    The RTFormer implementation based on PaddlePaddle.

    The original article refers to "Wang, Jian, Chenhui Gou, Qiman Wu, Haocheng Feng, 
    Junyu Han, Errui Ding, and Jingdong Wang. RTFormer: Efficient Design for Real-Time
    Semantic Segmentation with Transformer. arXiv preprint arXiv:2210.07124 (2022)."

    Args:
        num_classes (int): The unique number of target classes.
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        num_heads (int, optional): The num of heads in EABlock. Default: 8
        head_channels (int, optional): The channels of head in EABlock. Default: 128
        drop_rate (float, optional): The drop rate in EABlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_aux_head (bool, optional): Whether use auxiliary head. Default: True
        use_injection (list[boo], optional): Whether use injection in layer 4 and 5.
            Default: [True, True]
        lr_mult (float, optional): The multiplier of lr for DAPPM and head module. Default: 10
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
        in_channels (int, optional): The channels of input image. Default: 3
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_head=True,
                 use_injection=[True, True],
                 lr_mult=10.,
                 cross_size=12,
                 in_channels=3,
                 pretrained=None):
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels, base_chs, kernel_size=3, stride=2,
                      padding=1),
            bn2d(base_chs),
            nn.ReLU(),
            nn.Conv2D(base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(BasicBlock,
                                       base_chs,
                                       base_chs * 2,
                                       layer_nums[1],
                                       stride=2)
        self.layer3 = self._make_layer(BasicBlock,
                                       base_chs * 2,
                                       base_chs * 4,
                                       layer_nums[2],
                                       stride=2)
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2,
                                        1)
        self.compression3 = nn.Sequential(
            bn2d(base_chs * 4),
            nn.ReLU(),
            conv2d(base_chs * 4, base_chs * 2, kernel_size=1),
        )
        self.layer4 = EABlock(in_channels=[base_chs * 2, base_chs * 4],
                              out_channels=[base_chs * 2, base_chs * 8],
                              num_heads=num_heads,
                              drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate,
                              use_injection=use_injection[0],
                              use_cross_kv=True,
                              cross_size=cross_size)
        self.layer5 = EABlock(in_channels=[base_chs * 2, base_chs * 8],
                              out_channels=[base_chs * 2, base_chs * 8],
                              num_heads=num_heads,
                              drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate,
                              use_injection=use_injection[1],
                              use_cross_kv=True,
                              cross_size=cross_size)

        self.spp = DAPPM(base_chs * 8,
                         spp_channels,
                         base_chs * 2,
                         lr_mult=lr_mult)
        self.seghead = SegHead(base_chs * 4,
                               int(head_channels * 2),
                               num_classes,
                               lr_mult=lr_mult)
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.seghead_extra = SegHead(base_chs * 2,
                                         head_channels,
                                         num_classes,
                                         lr_mult=lr_mult)

        self.pretrained = pretrained
        self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.layer3_.apply(self._init_weights_kaiming)
        self.compression3.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)
        self.seghead.apply(self._init_weights_kaiming)
        if self.use_aux_head:
            self.seghead_extra.apply(self._init_weights_kaiming)

        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                bn2d(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(
            self.compression3(x3), size=x2.shape[2:], mode='bilinear')
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4([self.relu(x3_),
                               self.relu(x3)])  # 2c, 1/8; 8c, 1/16
        x5_, x5 = self.layer5([self.relu(x4_),
                               self.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(x6, size=x5_.shape[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(paddle.concat([x5_, x6], axis=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.training and self.use_aux_head:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(logit,
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=False) for logit in logit_list
        ]

        return logit_list


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           bias_attr=False,
           lr_mult=1.0,
           **kwargs):
    assert bias_attr in [True, False], "bias_attr should be True or False"
    weight_attr = paddle.ParamAttr(learning_rate=lr_mult)
    if bias_attr:
        bias_attr = paddle.ParamAttr(learning_rate=lr_mult)
    return nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     weight_attr=weight_attr,
                     bias_attr=bias_attr,
                     **kwargs)


def bn2d(in_channels, bn_mom=0.1, lr_mult=1.0, **kwargs):
    assert 'bias_attr' not in kwargs, "bias_attr must not in kwargs"
    param_attr = paddle.ParamAttr(learning_rate=lr_mult)
    return nn.BatchNorm2D(in_channels,
                          momentum=bn_mom,
                          weight_attr=param_attr,
                          bias_attr=param_attr,
                          **kwargs)


class BasicBlock(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out if self.no_relu else self.relu(out)


class MLP(nn.Layer):

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = bn2d(in_channels, epsilon=1e-06)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Layer):
    """
    The ExternalAttention implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = bn2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = self.create_parameter(
                shape=(inter_channels, in_channels, 1, 1),
                default_initializer=paddle.nn.initializer.Normal(std=0.001))
            self.v = self.create_parameter(
                shape=(out_channels, inter_channels, 1, 1),
                default_initializer=paddle.nn.initializer.Normal(std=0.001))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, value=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.)
            constant_init(m.bias, value=.0)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, value=0.)

    def _act_sn(self, x):
        x = x.reshape([-1, self.inter_channels, 0, 0]) * (self.inter_channels**
                                                          -0.5)
        x = F.softmax(x, axis=1)
        x = x.reshape([1, -1, 0, 0])
        return x

    def _act_dn(self, x):
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [0, self.num_heads, self.inter_channels // self.num_heads, -1])
        x = F.softmax(x, axis=3)
        x = x / (paddle.sum(x, axis=2, keepdim=True) + 1e-06)
        x = x.reshape([0, self.inter_channels, h, w])
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor. 
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(x,
                         self.k,
                         bias=None,
                         stride=2 if not self.same_in_out_chs else 1,
                         padding=0)  # n,c_in,h,w -> n,c_inter,h,w
            x = self._act_dn(x)  # n,c_inter,h,w
            x = F.conv2d(x, self.v, bias=None, stride=1,
                         padding=0)  # n,c_inter,h,w -> n,c_out,h,w
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should no be None when use_cross_kv"
            B = x.shape[0]
            assert B > 0, "The first dim of x ({}) should be greater than 0, please set input_shape for export.py".format(
                B)
            x = x.reshape([1, -1, 0, 0])  # n,c_in,h,w -> 1,n*c_in,h,w
            x = F.conv2d(x, cross_k, bias=None, stride=1, padding=0,
                         groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
            x = self._act_sn(x)
            x = F.conv2d(x, cross_v, bias=None, stride=1, padding=0,
                         groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
            x = x.reshape([-1, self.in_channels, 0,
                           0])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)
        return x


class EABlock(nn.Layer):
    """
    The EABlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_injection (bool, optional): Whether inject the high feature into low feature. Default: True
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: True
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_channels_l),
                conv2d(in_channels_l, out_channels_l, 1, 2, 0))
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_l = ExternalAttention(in_channels_l,
                                        out_channels_l,
                                        inter_channels=out_channels_l,
                                        num_heads=num_heads,
                                        use_cross_kv=False)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else Identity()

        # compression
        self.compression = nn.Sequential(
            bn2d(out_channels_l), nn.ReLU(),
            conv2d(out_channels_l, out_channels_h, kernel_size=1))
        self.compression.apply(self._init_weights_kaiming)

        # high resolution
        self.attn_h = ExternalAttention(in_channels_h,
                                        in_channels_h,
                                        inter_channels=cross_size * cross_size,
                                        num_heads=num_heads,
                                        use_cross_kv=use_cross_kv)
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                bn2d(out_channels_l),
                nn.AdaptiveMaxPool2D(output_size=(self.cross_size,
                                                  self.cross_size)),
                conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0))
            self.cross_kv.apply(self._init_weights)

        # injection
        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_channels_h),
                nn.ReLU(),
                conv2d(out_channels_h,
                       out_channels_l // 2,
                       kernel_size=3,
                       stride=2,
                       padding=1),
                bn2d(out_channels_l // 2),
                nn.ReLU(),
                conv2d(out_channels_l // 2,
                       out_channels_l,
                       kernel_size=3,
                       stride=2,
                       padding=1),
            )
            self.down.apply(self._init_weights_kaiming)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def forward(self, x):
        x_h, x_l = x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))  # n,out_chs_l,h,w

        # compression
        x_h_shape = x_h.shape[2:]
        x_l_cp = self.compression(x_l)
        x_h += F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_k, cross_v = paddle.split(cross_kv, 2, axis=1)
            cross_k = cross_k.transpose([0, 2, 3, 1]).reshape(
                [-1, self.out_channels_h, 1, 1])  # n*144,out_channels_h,1,1
            cross_v = cross_v.reshape(
                [-1, self.cross_size * self.cross_size, 1,
                 1])  # n*out_channels_h,144,1,1
            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k,
                                                   cross_v))  # n,out_chs_h,h,w

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Layer):

    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(kernel_size=5, stride=2, padding=2, exclusive=False),
            bn2d(in_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1, lr_mult=lr_mult))
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=9, stride=4, padding=4, exclusive=False),
            bn2d(in_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1, lr_mult=lr_mult))
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(kernel_size=17, stride=8, padding=8, exclusive=False),
            bn2d(in_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1, lr_mult=lr_mult))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)), bn2d(in_channels, lr_mult=lr_mult),
            nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1, lr_mult=lr_mult))
        self.scale0 = nn.Sequential(
            bn2d(in_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1, lr_mult=lr_mult))
        self.process1 = nn.Sequential(
            bn2d(inter_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(inter_channels,
                   inter_channels,
                   kernel_size=3,
                   padding=1,
                   lr_mult=lr_mult))
        self.process2 = nn.Sequential(
            bn2d(inter_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(inter_channels,
                   inter_channels,
                   kernel_size=3,
                   padding=1,
                   lr_mult=lr_mult))
        self.process3 = nn.Sequential(
            bn2d(inter_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(inter_channels,
                   inter_channels,
                   kernel_size=3,
                   padding=1,
                   lr_mult=lr_mult))
        self.process4 = nn.Sequential(
            bn2d(inter_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(inter_channels,
                   inter_channels,
                   kernel_size=3,
                   padding=1,
                   lr_mult=lr_mult))
        self.compression = nn.Sequential(
            bn2d(inter_channels * 5, lr_mult=lr_mult), nn.ReLU(),
            conv2d(inter_channels * 5,
                   out_channels,
                   kernel_size=1,
                   lr_mult=lr_mult))
        self.shortcut = nn.Sequential(
            bn2d(in_channels, lr_mult=lr_mult), nn.ReLU(),
            conv2d(in_channels, out_channels, kernel_size=1, lr_mult=lr_mult))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (F.interpolate(self.scale1(x), size=x_shape, mode='bilinear') +
                 x_list[0])))
        x_list.append((self.process2(
            (F.interpolate(self.scale2(x), size=x_shape, mode='bilinear') +
             x_list[1]))))
        x_list.append(
            self.process3(
                (F.interpolate(self.scale3(x), size=x_shape, mode='bilinear') +
                 x_list[2])))
        x_list.append(
            self.process4(
                (F.interpolate(self.scale4(x), size=x_shape, mode='bilinear') +
                 x_list[3])))

        out = self.compression(paddle.concat(x_list, axis=1)) + self.shortcut(x)
        return out


class SegHead(nn.Layer):

    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        super().__init__()
        self.bn1 = bn2d(in_channels, lr_mult=lr_mult)
        self.conv1 = conv2d(in_channels,
                            inter_channels,
                            kernel_size=3,
                            padding=1,
                            lr_mult=lr_mult)
        self.bn2 = bn2d(inter_channels, lr_mult=lr_mult)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(inter_channels,
                            out_channels,
                            kernel_size=1,
                            padding=0,
                            bias_attr=True,
                            lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out
