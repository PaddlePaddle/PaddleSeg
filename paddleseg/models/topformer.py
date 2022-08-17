# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class TopFormer(nn.Layer):
    """
    The Token Pyramid Transformer(TopFormer) implementation based on PaddlePaddle.

    The original article refers to
    Zhang, Wenqiang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu,
    and Chunhua Shen. "TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation." 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    pp. 12083-12093. 2022.

    Args:
        num_classes(int,optional): The unique number of target classes.
        backbone(nn.Layer): Backbone network.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=False,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone

        head_in_channels = [
            i for i in backbone.injection_out_channels if i is not None
        ]
        self.head = TopFormerHead(
            num_classes=num_classes,
            in_channels=head_in_channels,
            use_dw=head_use_dw,
            align_corners=align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        x = self.backbone(x)  # len=3, 1/8,1/16,1/32
        x = self.head(x)
        x = F.interpolate(
            x, x_hw, mode='bilinear', align_corners=self.align_corners)
        return [x]


def resize(input_data,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input_data.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and
                     input_w > 1) and (output_h - 1) % (input_h - 1) and
                    (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """

    def forward(self, inputs):
        return inputs


class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.SyncBatchNorm,
                 act=nn.ReLU,
                 bias_attr=False):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias_attr)
        self.act = act() if act is not None else Identity()
        self.norm = norm(out_channels) if norm is not None else Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TopFormerHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 in_index=[0, 1, 2],
                 in_transform='multiple_select',
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()

        self.in_index = in_index
        self.in_transform = in_transform
        self.align_corners = align_corners

        self._init_inputs(in_channels, in_index, in_transform)
        self.linear_fuse = ConvModule(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1, )
        self.dropout = nn.Dropout2D(dropout_ratio)
        self.conv_seg = nn.Conv2D(
            self.last_channels, num_classes, kernel_size=1)

    def _init_inputs(self, in_channels, in_index, in_transform):
        assert in_transform in [None, 'resize_concat', 'multiple_select']
        if in_transform is not None:
            assert len(in_channels) == len(in_index)
            if in_transform == 'resize_concat':
                self.last_channels = sum(in_channels)
            else:
                self.last_channels = in_channels[0]
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.last_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.in_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            inputs = [
                resize(
                    input_data=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = paddle.concat(inputs, axis=1)
        elif self.in_transform == 'multiple_select':
            inputs_tmp = [inputs[i] for i in self.in_index]
            inputs = inputs_tmp[0]
            for x in inputs_tmp[1:]:
                x = resize(
                    x,
                    size=inputs.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                inputs += x
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x
