# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

__all__ = ["TopTransformer_Base", "TopTransformer_Small", "TopTransformer_Tiny"]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mlp(nn.Layer):
    """ MLP module"""

    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        #w_attr_1, b_attr_1 = self._init_weights_linear()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = ConvNormAct(embed_dim, hidden_dim, kernel_size=1, act=None)
        self.dwconv = nn.Conv2D(
            hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = ConvNormAct(hidden_dim, embed_dim, kernel_size=1, act=None)
        self.act = nn.ReLU6()
        self.dropout = nn.Dropout(dropout)

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self, embed_dim, key_dim, num_heads, attn_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_size = key_dim
        self.all_head_size = self.attn_head_size * num_heads
        self.dh = int(self.attn_head_size * attn_ratio) * num_heads

        self.q = ConvNormAct(
            embed_dim, self.all_head_size, kernel_size=1, act=None)
        self.k = ConvNormAct(
            embed_dim, self.all_head_size, kernel_size=1, act=None)
        self.v = ConvNormAct(embed_dim, self.dh, kernel_size=1, act=None)

        self.scales = self.attn_head_size**-0.5

        self.proj = nn.Sequential(*[
            nn.ReLU6(), ConvNormAct(
                self.dh, self.embed_dim, kernel_size=1, act=None)
        ])

        self.softmax = nn.Softmax(-1)

    def transpose_multihead(self, x):
        # in_shape: [batch_size, all_head_size, H', W']
        N, C, H, W = x.shape
        x = x.reshape([N, self.num_heads, -1, H, W])
        x = x.flatten(-2)  # [N, num_heads, attn_head_size, H*W]
        x = x.transpose([0, 1, 3, 2])  #[N, num_heads, H*W, attn_head_size]
        return x

    def forward(self, x):
        N, C, H, W = x.shape
        q = self.q(x)  # N, C, H', W'
        q = self.transpose_multihead(q)
        k = self.k(x)
        k = self.transpose_multihead(k)
        v = self.v(x)  # 
        v = self.transpose_multihead(v)

        #q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        #attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 1, 3, 2])
        z = z.reshape([N, self.dh, H, W])
        z = self.proj(z)
        return z


class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 key_dim,
                 num_heads=8,
                 mlp_ratio=2.0,
                 attn_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()

        #self.attn_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(embed_dim, key_dim, num_heads, attn_ratio)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        #self.mlp_norm = nn.LayerNorm(embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        #x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        #x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class Transformer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 key_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 attn_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(
                EncoderLayer(embed_dim, key_dim, num_heads, mlp_ratio,
                             attn_ratio, dropout, attention_dropout, droppath))
        self.layers = nn.LayerList(layer_list)

        #w_attr_1, b_attr_1 = _init_weights_layernorm()
        #self.norm = nn.LayerNorm(embed_dim,
        #                         weight_attr=w_attr_1,
        #                         bias_attr=b_attr_1,
        #                         epsilon=1e-6)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        #out = self.norm(x)
        #return out
        return x


class ConvNormAct(nn.Layer):
    """Layer ops: Conv2D -> SyncBatchNorm -> ReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.ReLU(),
                 norm=nn.SyncBatchNorm):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = Identity() if norm is None else norm(out_channels)
        self.act = Identity() if act is None else act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class MobileV2Block(nn.Layer):
    """Mobilenet v2 InvertedResidual block, hacked from torchvision"""

    def __init__(self, inp, oup, kernel_size=3, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            layers.append(ConvNormAct(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvNormAct(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
                padding=kernel_size // 2),
            # pw-linear
            nn.Conv2D(
                hidden_dim, oup, 1, 1, 0, bias_attr=False),
            nn.SyncBatchNorm(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class TokenPyramidModule(nn.Layer):
    def __init__(self, cfgs, out_indices, input_channel=16, width_mult=1.):
        super().__init__()
        self.out_indices = out_indices
        self.stem = ConvNormAct(
            3, input_channel, kernel_size=3, stride=2, padding=1)

        self.layers = nn.LayerList()
        for idx, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            expand_size = t * input_channel
            expand_size = _make_divisible(expand_size * width_mult, 8)
            self.layers.append(
                MobileV2Block(
                    input_channel,
                    output_channel,
                    kernel_size=k,
                    stride=s,
                    expansion=t))
            input_channel = output_channel

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class PyramidPoolAgg(nn.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        N, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return paddle.concat(
            [nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs],
            axis=1)


class InjectionMultiSum(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.global_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.global_act = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.act = nn.Hardsigmoid()

    def forward(self, x_local, x_global):
        N, C, H, W = x_local.shape

        local_feature = self.local_embedding(x_local)

        global_act = self.global_act(x_global)
        global_act = self.act(global_act)
        global_act = nn.functional.interpolate(
            global_act, size=(H, W), mode='bilinear', align_corners=False)
        sigmoid_act = F.interpolate(
            self.act(global_act),
            size=(H, W),
            mode='bilinear',
            align_corners=False)

        global_feature = self.global_embedding(x_global)
        global_feature = nn.functional.interpolate(
            global_feature, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feature * sigmoid_act + global_feature
        return out


class InjectionMultiSumCBR(InjectionMultiSum):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.local_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1)
        self.global_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1)
        self.global_act = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None, norm=None)


class FuseBlockSum(nn.Layer):
    def __init__(self, in_channels, out_channels, act=nn.ReLU6()):
        super().__init__()
        self.local_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.global_embedding = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=None)
        self.act = Identity() if act is None else act

    def forward_features(self, x_local, x_global):
        N, C, H, W = x_local.shape

        local_feature = self.local_embedding(x_local)

        global_feature = self.global_embedding(x_global)
        global_feature = self.act(global_feature)
        global_feature = nn.functional.interpolate(
            global_feature, size=(H, W), mode='bilinear', align_corners=False)

        return local_feature, global_feature

    def forward(self, x_local, x_global):
        local_features, global_features = self.forward_features(x_local,
                                                                x_global)
        out = local_features + global_features
        return out


class FuseBlockMulti(nn.Layer):
    def __init__(self, in_channels, out_channels, act=nn.Hardsigmoid()):
        super().__init__(in_channels, out_channels, act)

    def forward(self, x_local, x_global):
        local_features, global_features = self.forward_features(x_local,
                                                                x_global)
        out = local_features * global_features
        return out


class TopTransformer(nn.Layer):
    """
    The Token Pyramid Transformer(TopFormer) implementation based on PaddlePaddle.

    The original article refers to
    Zhang, Wenqiang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu,
    and Chunhua Shen. "TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation." 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    pp. 12083-12093. 2022.

    Args:
        cfgs (List): The config of backbone.
        input_channels (List): The input channels.
        out_channels (List): The output channels.
        embed_out_indice (List): xxxxxx 
        .... (todo)
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 cfgs,
                 input_channels,
                 out_channels,
                 embed_out_indice,
                 decode_out_indices=[1, 2, 3],
                 depth=4,
                 key_dim=16,
                 num_heads=8,
                 attn_ratio=2,
                 mlp_ratio=2,
                 dropout=0,
                 attention_dropout=0,
                 drop_path=0,
                 c2t_stride=2,
                 injection_type="multi_sum",
                 injection=True,
                 pretrained=None):
        super().__init__()

        self.input_channels = input_channels
        self.injection = injection
        self.embed_dim = sum(self.input_channels)
        self.decode_out_indices = decode_out_indices

        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        self.trans = Transformer(
            embed_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            attn_ratio=attn_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            droppath=drop_path)

        self.sim = nn.LayerList()
        sim_block_dict = {
            "fuse_sum": FuseBlockSum,
            "fuse_multi": FuseBlockMulti,
            "multi_sum": InjectionMultiSum,
            "multi_sim_cbr": InjectionMultiSumCBR
        }
        sim_block = sim_block_dict[injection_type]
        if self.injection:
            for idx, (channel, out_channel
                      ) in enumerate(zip(self.input_channels, out_channels)):
                if idx in self.decode_out_indices:
                    self.sim.append(sim_block(channel, out_channel))
                else:
                    self.sim.append(Identity())

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        outputs = self.tpm(x)
        out = self.ppa(outputs)
        out = self.trans(out)

        if self.injection:
            xx = out.split(
                self.input_channels, axis=1)  # self.input_channels is a list
            results = []
            for i in range(len(self.input_channels)):
                if i in self.decode_out_indices:
                    local_tokens = outputs[i]
                    global_semantics = xx[i]
                    out_ = self.sim[i](local_tokens, global_semantics)
                    results.append(out_)
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

    model = TopTransformer(
        cfgs=cfgs,
        input_channels=[32, 64, 128, 160],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        depth=4,
        key_dim=16,
        num_heads=8,
        attn_ratio=2,
        mlp_ratio=2,
        dropout=0,
        attention_dropout=0,
        drop_path=0,
        c2t_stride=2,
        injection_type="multi_sum",
        injection=True,
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
        input_channels=[24, 48, 96, 128],
        out_channels=[None, 192, 192, 192],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        depth=4,
        key_dim=16,
        num_heads=6,
        attn_ratio=2,
        mlp_ratio=2,
        dropout=0,
        attention_dropout=0,
        drop_path=0,
        c2t_stride=2,
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
        input_channels=[16, 32, 64, 96],
        out_channels=[None, 128, 128, 128],
        embed_out_indice=[2, 4, 6, 8],
        decode_out_indices=[1, 2, 3],
        depth=4,
        key_dim=16,
        num_heads=4,
        attn_ratio=2,
        mlp_ratio=2,
        dropout=0,
        attention_dropout=0,
        drop_path=0,
        c2t_stride=2,
        injection_type="multi_sum",
        injection=True,
        **kwargs)
    return model
