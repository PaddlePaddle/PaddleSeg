# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy

import paddle
from paddle import create_parameter
import paddle.nn as nn
from paddle.nn.functional import upsample
from paddle.nn.functional import interpolate

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init


class BaseNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[0, 1, 2, 3],
                 aux=True,
                 jpu=True,
                 norm_layer=nn.BatchNorm2D,
                 pretrained=None):
        super().__init__()
        self.nclass = num_classes
        self.aux = aux
        self.jpu = jpu

        self.backbone = backbone
        self.backbone_indices = backbone_indices

        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.jpu = JPU([512, 1024, 2048],
                       width=512,
                       norm_layer=norm_layer,
                       up_kwargs=self._up_kwargs) if jpu else None

    def base_forward(self, x):
        feat_list = self.backbone(x)
        c1, c2, c3, c4 = [feat_list[i] for i in self.backbone_indices]

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4


@manager.MODELS.add_component
class GINet(BaseNet):
    """
    The GINet implementation based on PaddlePaddle.

    The original article refers to
    Wu, Tianyi, Yu Lu, Yu Zhu, Chuang Zhang, Ming Wu, Zhanyu Ma, and Guodong Guo. "GINet: Graph interaction network for scene parsing." In European Conference on Computer Vision, pp. 34-51. Springer, Cham, 2020.
    (https://arxiv.org/pdf/2009.06160).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        aux (bool, optional)): whether to use another branch to optimize. Default:True.
        jpu (bool, optional)): whether to use jpu unit in the base forward. Default:True.
        norm_layer(Paddle.nn.Layer, optional)): The kind of normalization in the network. Default: nn.BatchNorm2D.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[0, 1, 2, 3],
                 aux=True,
                 jpu=True,
                 norm_layer=nn.BatchNorm2D,
                 pretrained=None):
        super(GINet, self).__init__(num_classes, backbone, backbone_indices,
                                    aux, jpu, norm_layer, pretrained)

        self.head = GIHead(
            in_channels=2048, nclass=num_classes, up_kwargs=self._up_kwargs)

        if self.aux:
            self.auxlayer = FCNHead(1024, num_classes, norm_layer)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        _, _, h, w = x.shape
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x, _ = self.head(c4)
        x = upsample(x, (h, w), **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h, w), **self._up_kwargs)

            outputs.append(auxout)

        return outputs

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class FCNHead(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(
            nn.Conv2D(
                in_channels, inter_channels, 3, padding=1, bias_attr=False),
            norm_layer(inter_channels), nn.ReLU(), nn.Dropout2D(0.1),
            nn.Conv2D(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class GIHead(nn.Layer):
    def __init__(self, in_channels, nclass, up_kwargs):
        super().__init__()
        self.nclass = nclass
        inter_channels = in_channels // 4
        self.inp = paddle.zeros(shape=(nclass, 300), dtype='float32')
        self.inp = create_parameter(
            shape=self.inp.shape,
            dtype=str(self.inp.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.inp))

        self.fc1 = nn.Sequential(
            nn.Linear(300, 128), nn.BatchNorm1D(128), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1D(256), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2D(
                in_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False), nn.BatchNorm2D(inter_channels), nn.ReLU())
        self.gloru = Global_Reason_Unit(
            in_channels=inter_channels,
            num_state=256,
            num_node=84,
            nclass=nclass)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1), nn.Conv2D(inter_channels, nclass, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        inp = self.inp.detach()

        inp = self.fc1(inp)  ## almost all zeros input into fc
        inp = self.fc2(inp).unsqueeze(axis=0).transpose((0, 2, 1))\
                           .expand((B, 256, self.nclass))

        out = self.conv5(x)

        out, se_out = self.gloru(out, inp)
        out = self.conv6(out)
        return out, se_out


class Global_Reason_Unit(nn.Layer):
    """
        Graph G = (V, \sigma), |V| is the number of vertext,
        denoted as num_node, each vertex feature is d-dim vector,
        d is equal to num_state
        Reference:
            Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks" (https://arxiv.org/abs/1811.12814)
    """

    def __init__(self, in_channels, num_state=256, num_node=84, nclass=59):
        super().__init__()
        self.num_state = num_state
        self.conv_theta = nn.Conv2D(
            in_channels, num_node, kernel_size=1, stride=1, padding=0)
        self.conv_phi = nn.Conv2D(
            in_channels, num_state, kernel_size=1, stride=1, padding=0)
        self.graph = Graph_layer(num_state, num_node, nclass)
        self.extend_dim = nn.Conv2D(
            num_state, in_channels, kernel_size=1, bias_attr=False)


        weight_attr = paddle.ParamAttr(name="weight", initializer=\
                            paddle.nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(name="bias", initializer=\
                            paddle.nn.initializer.Constant(value=0.0))
        self.bn = nn.BatchNorm2D(
            in_channels, weight_attr=weight_attr, bias_attr=bias_attr)

    def forward(self, x, inp):
        """
            input: x shape is B x C x H x W
            output: x_new shape is B x C x H x W
        """
        B = self.conv_theta(x)
        sizeB = B.shape
        B = B.reshape((sizeB[0], sizeB[1], -1))

        sizex = x.shape
        x_reduce = self.conv_phi(x)
        x_reduce = x_reduce.reshape((sizex[0], -1, sizex[2]*sizex[3]))\
                           .transpose((0, 2, 1))

        V = paddle.bmm(B, x_reduce).transpose((0, 2, 1))
        V = paddle.divide(
            V, paddle.to_tensor([sizex[2] * sizex[3]], dtype='float32'))

        class_node, new_V = self.graph(inp, V)
        D = B.reshape((sizeB[0], -1, sizeB[2] * sizeB[3])).transpose((0, 2, 1))
        Y = paddle.bmm(D, new_V.transpose((0, 2, 1)))
        Y = Y.transpose((0, 2, 1)).reshape((sizex[0], self.num_state, \
                                            sizex[2], -1))
        Y = self.extend_dim(Y)
        Y = self.bn(Y)
        out = Y + x

        return out, class_node


class Graph_layer(nn.Layer):
    def __init__(self, num_state, num_node, num_class):
        super().__init__()
        self.vis_gcn = GCN(num_state, num_node)
        self.word_gcn = GCN(num_state, num_class)
        self.transfer = Graph_transfer(num_state)
        self.gamma_vis = paddle.zeros([num_node])
        self.gamma_word = paddle.zeros([num_class])
        self.gamma_vis = create_parameter(
            shape=self.gamma_vis.shape,
            dtype=str(self.gamma_vis.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.gamma_vis))
        self.gamma_word = create_parameter(
            shape=self.gamma_word.shape,
            dtype=str(self.gamma_word.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.gamma_word))

    def forward(self, inp, vis_node):
        inp = self.word_gcn(inp)
        new_V = self.vis_gcn(vis_node)
        class_node, vis_node = self.transfer(inp, new_V)

        class_node = self.gamma_word * inp + class_node
        new_V = self.gamma_vis * vis_node + new_V
        return class_node, new_V


class GCN(nn.Layer):
    def __init__(self, num_state=128, num_node=64, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1D(
            num_node,
            num_node,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1D(
            num_state,
            num_state,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias_attr=bias)

    def forward(self, x):
        h = self.conv1(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class Graph_transfer(nn.Layer):
    """
        Transfer vis graph to class node
        Transfer class node to vis feature
    """

    def __init__(self, in_dim):
        super().__init__()
        self.channle_in = in_dim
        self.query_conv = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv_vis = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_word = nn.Conv1D(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax_vis = nn.Softmax(axis=-1)
        self.softmax_word = nn.Softmax(axis=-2)

    def forward(self, word, vis_node):
        """
            inputs :
                word : input feature maps( B X C X Nclass)
                vis_node: B, C, num_node
            returns :
        """
        m_batchsize, C, Nc = word.shape
        m_batchsize, C, Nn = vis_node.shape

        proj_query = self.query_conv(word).reshape((m_batchsize, -1, Nc))\
                                          .transpose((0, 2, 1))
        proj_key = self.key_conv(vis_node).reshape((m_batchsize, -1, Nn))

        energy = paddle.bmm(proj_query, proj_key)
        attention_vis = self.softmax_vis(energy).transpose((0, 2, 1))
        attention_word = self.softmax_word(energy)

        proj_value_vis = self.value_conv_vis(vis_node).reshape(
            (m_batchsize, -1, Nn))  #B, C, num_node
        proj_value_word = self.value_conv_word(word).reshape(
            (m_batchsize, -1, Nc))  #B, C, num_class

        class_out = paddle.bmm(proj_value_vis, attention_vis)  #B, C, nclass
        node_out = paddle.bmm(proj_value_word, attention_word)  #B, C, node
        return class_out, node_out


class JPU(nn.Layer):
    def __init__(self,
                 in_channels,
                 width=512,
                 norm_layer=nn.BatchNorm2D,
                 **kwargs):
        super().__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels[-3], width, 3, padding=1, bias_attr=False),
            norm_layer(width), nn.ReLU())

        self.dilation1 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, 3, padding=1, dilation=1, bias=False),
            norm_layer(width), nn.ReLU())
        self.dilation2 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, 3, padding=2, dilation=2, bias=False),
            norm_layer(width), nn.ReLU())
        self.dilation3 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, 3, padding=4, dilation=4, bias=False),
            norm_layer(width), nn.ReLU())
        self.dilation4 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, 3, padding=8, dilation=8, bias=False),
            norm_layer(width), nn.ReLU())

    def forward(self, *inputs):
        feats = [
            self.conv5(inputs[-1]),
            self.conv4(inputs[-2]),
            self.conv3(inputs[-3])
        ]
        size = feats[-1].shape[2:]
        feats[-2] = interpolate(
            feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = interpolate(
            feats[-3], size, mode='bilinear', align_corners=True)

        feat = paddle.concat(feats, axis=1)
        feat = paddle.concat([
            self.dilation1(feat),
            self.dilation2(feat),
            self.dilation3(feat),
            self.dilation4(feat)
        ],
                             axis=1)

        return inputs[0], inputs[1], inputs[2], feat


class SeparableConv2d(nn.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 norm_layer=nn.BatchNorm2D):
        super().__init__()

        self.conv1 = nn.Conv2D(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=inplanes,
            bias_attr=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2D(
            inplanes, planes, 1, 1, 0, 1, 1, bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
