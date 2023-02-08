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

__all__ = ["TopTransformer_Base", "TopTransformer_Small", "TopTransformer_Tiny"]


def make_divisible(val, divisor, min_value=None):
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


class TokenPyramidModule(nn.Layer):
    def __init__(self,
                 cfgs,
                 out_indices,
                 in_channels=3,
                 inp_channel=16,
                 activation=nn.ReLU,
                 width_mult=1.,
                 lr_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2DBN(
                in_channels, inp_channel, 3, 2, 1, lr_mult=lr_mult),
            activation())

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = make_divisible(exp_size * width_mult, 8)
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
            self.layers.append(layer_name)
            inp_channel = output_channel

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


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
        x_shape = paddle.shape(x)  # NCHW
        H, W = x_shape[2], x_shape[3]

        qq = self.to_q(x).reshape(
            [0, self.num_heads, self.key_dim, -1]).transpose(
                [0, 1, 3, 2])  # [N , nh,  HW, dim]
        kk = self.to_k(x).reshape(
            [0, self.num_heads, self.key_dim, -1])  # [N , nh, dim, HW]
        vv = self.to_v(x).reshape([0, self.num_heads, self.d, -1]).transpose(
            [0, 1, 3, 2])  # [N , nh, HW, dh, ]

        attn = paddle.matmul(qq, kk)  # HW2 # [N , nh, HW ,HW] scale
        attn = F.softmax(attn, axis=-1)

        xx = paddle.matmul(attn, vv)  # [N , nh, HW, dh]

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

        # from paddleseg.models.rtformer import ExternalAttention 
        # self.attn = ExternalAttention(dim, dim, 256, num_heads=8, use_cross_kv=False) 
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


class InjectionMultiSum(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activations=None,
                 global_in_channels=None,
                 lr_mult=1.0):
        super(InjectionMultiSum, self).__init__()

        self.local_embedding = ConvBNAct(
            in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.global_embedding = ConvBNAct(
            global_in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.global_act = ConvBNAct(
            global_in_channels, out_channels, kernel_size=1, lr_mult=lr_mult)
        self.act = HSigmoid()

    def forward(self, x_low, x_global):
        xl_hw = paddle.shape(x_low)[2:]
        local_feat = self.local_embedding(x_low)

        global_act = self.global_act(x_global)
        sig_act = F.interpolate(
            self.act(global_act), xl_hw, mode='bilinear', align_corners=False)

        global_feat = self.global_embedding(x_global)
        global_feat = F.interpolate(
            global_feat, xl_hw, mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
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

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64
        low_feat1 = F.interpolate(inputs[0], scale_factor=0.5, mode="bilinear")
        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2_act = self.act_list[1](self.act_embedding_list[1](inputs[1]))
        low_feat2 = self.embedding_list[1](inputs[1])

        low_feat3_act = F.interpolate(
            self.act_list[2](self.act_embedding_list[2](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        low_feat3 = F.interpolate(
            self.embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        high_feat_act = F.interpolate(
            self.act_list[3](self.act_embedding_list[3](inputs[3])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[3](inputs[3]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat1_act * low_feat2_act * low_feat3_act * high_feat_act * (
            low_feat1 + low_feat2 + low_feat3) + high_feat

        return res


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,
    "multi_sum": InjectionMultiSum,
    "multi_sum_cbr": InjectionMultiSumCBR,
}


class TopTransformer(nn.Layer):
    def __init__(self,
                 cfgs,
                 injection_out_channels,
                 encoder_out_indices,
                 trans_out_indices=[1, 2, 3],
                 depths=4,
                 key_dim=16,
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=2,
                 c2t_stride=2,
                 drop_path_rate=0.,
                 act_layer=nn.ReLU6,
                 injection_type="InjectionMultiSumallmultiallsum",
                 injection=True,
                 lr_mult=1.0,
                 in_channels=3,
                 backbone='top_transformer',
                 pretrained=None):
        super().__init__()
        print('The backbone is ', backbone)
        if backbone == 'top_transformer':
            self.feat_channels = [
                c[2] for i, c in enumerate(cfgs) if i in encoder_out_indices
            ]
            self.tpm = TokenPyramidModule(
                cfgs=cfgs,
                out_indices=encoder_out_indices,
                in_channels=in_channels,
                lr_mult=lr_mult)
            pretrained = "saved_model/topformer74.9.pdparams"
        elif backbone == 'esnet':
            self.feat_channels = [24, 56, 120, 232]
            from .esnet import ESNet_x0_5
            self.tpm = ESNet_x0_5()
            pretrained = "saved_model/best_model_esnet.pdparams"
        elif backbone == 'esnet0_75':
            self.feat_channels = [24, 88, 176, 352]
            from .esnet import ESNet_x0_75
            self.tpm = ESNet_x0_75()
            pretrained = "saved_model/best_model_esnet075.pdparams"
        elif backbone == 'esnet1_0':
            self.feat_channels = [24, 120, 232, 464]
            from .esnet import ESNet_x1_0
            self.tpm = ESNet_x1_0()
            pretrained = None  # TODO add pretrain model
        elif backbone == 'mobilenetv3':
            self.feat_channels = [24, 40, 112, 160]
            from .mobilenetv3 import MobileNetV3_large_x1_0
            self.tpm = MobileNetV3_large_x1_0()
            pretrained = 'saved_model/best_model_mbv3.pdparams'
        elif backbone == 'mobilenetv3_large_075':
            self.feat_channels = [24, 32, 88, 120]
            from .mobilenetv3 import MobileNetV3_large_x0_75
            self.tpm = MobileNetV3_large_x0_75()
            pretrained = None
        elif backbone == 'mobilenetv3_small_125':
            self.feat_channels = [24, 32, 64, 120]
            from .mobilenetv3 import MobileNetV3_small_x1_25
            self.tpm = MobileNetV3_small_x1_25()
            pretrained = None
        elif backbone == 'mobilenetv3_edit':
            self.feat_channels = [32, 64, 128, 160]
            from .mobilenetv3 import MobileNetV3_large_x1_0_edit
            self.tpm = MobileNetV3_large_x1_0_edit()
            pretrained = 'saved_model/best_model_mbv3e.pdparams'
        elif backbone == 'MobileNetV3_large_x1_0_edit_x0_75':
            self.feat_channels = [24, 48, 96, 120]
            from .mobilenetv3 import MobileNetV3_large_x1_0_edit_x0_75
            self.tpm = MobileNetV3_large_x1_0_edit_x0_75()
            pretrained = 'saved_model/best_model_mbv3edit075.pdparams'
        elif backbone == "MobileNetV3_large_x1_25_edit":
            self.feat_channels = [40, 80, 160, 200]
            from .mobilenetv3 import MobileNetV3_large_x1_25_edit
            self.tpm = MobileNetV3_large_x1_25_edit()
            pretrained = 'saved_model/best_model_mbv3edit125.pdparams'
        elif backbone == "MobileNetV2":
            from .mobilenetv2 import MobileNetV2_x1_0
            self.tpm = MobileNetV2_x1_0()
            pretrained = None
            self.injection = False
        else:
            raise NotImplementedError('Backbone {} is not supported.'.format(
                backbone))

        self.injection_out_channels = injection_out_channels
        self.injection = injection
        self.embed_dim = sum(self.feat_channels)
        self.trans_out_indices = trans_out_indices
        self.injection_type = injection_type

        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in \
               paddle.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0,
            attn_drop=0,
            drop_path=dpr,
            act_layer=act_layer,
            lr_mult=lr_mult)

        if self.injection:
            if injection_type == 'InjectionMultiSumallmultiallsum':
                self.inj_module = InjectionMultiSumallmultiallsum(
                    in_channels=self.feat_channels[1:] + [self.embed_dim],
                    activations=act_layer,
                    lr_mult=lr_mult
                )  # TODO whether injection module is also backbone? set to 0.1?
                print('Using AAM')
            else:
                self.SIM = nn.LayerList()
                inj_module = SIM_BLOCK[injection_type]
                # inj_module = InjectionMultiSumSimple
                for i in range(len(self.feat_channels)):
                    if i in trans_out_indices:
                        self.SIM.append(
                            inj_module(
                                self.feat_channels[i],
                                injection_out_channels[i],
                                activations=act_layer,
                                global_in_channels=self.embed_dim,
                                lr_mult=lr_mult))
                    else:
                        self.SIM.append(Identity())

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        # [4, 3, 512, 512] [512,3,224,224] channels [32,64,128,160]
        outputs = self.tpm(
            x
        )  # 4 个输出 [4, 32, 128, 128] [4, 64, 64, 64] [4, 128, 32, 32][4, 160, 16, 16]        

        out = self.ppa(outputs)

        out = self.trans(out)

        if self.injection:
            if self.injection_type == 'InjectionMultiSumallmultiallsum':
                return self.inj_module(
                    [outputs[i] for i in self.trans_out_indices] + [out])
            else:
                # xx = out.split(self.feat_channels, axis=1)
                results = []
                for i in range(len(self.feat_channels)):
                    if i in self.trans_out_indices:
                        local_tokens = outputs[i]
                        global_semantics = out  #xx[i]
                        out_ = self.SIM[i](local_tokens, global_semantics)
                        results.append(out_)  # 64, 128, 256
                return results
        else:
            outputs.append(out)
            return outputs


@manager.BACKBONES.add_component
def TopTransformer_Base(**kwargs):
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
    prev_cfgs = [
        [3, 1, 16, 1],  # 1/2        
        [3, 6, 24, 2],  # 1/4 1    ->32  
        [3, 6, 24, 1],  # 1/4 1    !  
        [3, 6, 32, 2],  #            
        [3, 6, 32, 1],  #            
        [3, 6, 32, 1],  #          !  
        [3, 6, 64, 2],  # 1/8 3      
        [3, 6, 64, 1],  #            
        [3, 6, 64, 1],  #           ! 
        [3, 6, 64, 1],  #           ! 
        [3, 6, 96, 1],  # 1/16 5     
        [3, 6, 96, 1],  # 1/16 5     
        [3, 6, 96, 1],  # 1/16 5    ! ->128 
        [3, 6, 160, 2],  #            
        [3, 6, 160, 1],  #            
        [3, 6, 160, 1],  #            
        [3, 6, 320, 1],  # 1/32 7 !    
    ]

    model = TopTransformer(
        cfgs=cfgs,
        injection_out_channels=[None, 256, 256, 256],
        encoder_out_indices=[2, 4, 6, 9],
        trans_out_indices=[1, 2, 3],
        depths=4,
        key_dim=16,
        num_heads=8,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.,
        act_layer=nn.ReLU6,
        injection=True,
        injection_type="InjectionMultiSumallmultiallsum",
        **kwargs)
    return model


@manager.BACKBONES.add_component
def TopTransformer_Small(**kwargs):
    cfgs = [
        # k,  t,  c, s
        [3, 1, 16, 1],  # 1/2        
        [3, 4, 24, 2],  # 1/4 1      
        [3, 3, 24, 1],  #            
        [5, 3, 48, 2],  # 1/8 3      
        [5, 3, 48, 1],  #            
        [3, 3, 96, 2],  # 1/16 5     
        [3, 3, 96, 1],  #            
        [5, 6, 128, 2],  # 1/32 7     
        [5, 6, 128, 1],  #            
        [3, 6, 128, 1],  #           
    ]

    model = TopTransformer(
        cfgs=cfgs,
        injection_out_channels=[None, 192, 192, 192],
        encoder_out_indices=[2, 4, 6, 9],
        trans_out_indices=[1, 2, 3],
        depths=4,
        key_dim=16,
        num_heads=6,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.,
        act_layer=nn.ReLU6,
        injection_type="multi_sum",
        injection=True,
        **kwargs)
    return model


@manager.BACKBONES.add_component
def TopTransformer_Tiny(**kwargs):
    cfgs = [
        # k,  t,  c, s
        [3, 1, 16, 1],  # 1/2       
        [3, 4, 16, 2],  # 1/4 1      
        [3, 3, 16, 1],  #            
        [5, 3, 32, 2],  # 1/8 3      
        [5, 3, 32, 1],  #            
        [3, 3, 64, 2],  # 1/16 5     
        [3, 3, 64, 1],  #            
        [5, 6, 96, 2],  # 1/32 7     
        [5, 6, 96, 1],  #               
    ]

    model = TopTransformer(
        cfgs=cfgs,
        injection_out_channels=[None, 128, 128, 128],
        encoder_out_indices=[2, 4, 6, 8],
        trans_out_indices=[1, 2, 3],
        depths=4,
        key_dim=16,
        num_heads=4,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.,
        act_layer=nn.ReLU6,
        injection_type="multi_sum",
        injection=True,
        **kwargs)
    return model
