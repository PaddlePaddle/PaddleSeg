# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.utils import utils
from paddleseg.models import layers

__all__ = ["Xception41_deeplab", "Xception65_deeplab", "Xception71_deeplab"]


def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


def check_stride(s, os):
    if s <= os:
        return True
    else:
        return False


def check_points(count, points):
    if points is None:
        return False
    else:
        if isinstance(points, list):
            return (True if count in points else False)
        else:
            return (True if count == points else False)


def gen_bottleneck_params(backbone='xception_65'):
    if backbone == 'xception_65':
        bottleneck_params = {
            "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
            "middle_flow": (16, 1, 728),
            "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])
        }
    elif backbone == 'xception_41':
        bottleneck_params = {
            "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
            "middle_flow": (8, 1, 728),
            "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])
        }
    elif backbone == 'xception_71':
        bottleneck_params = {
            "entry_flow": (5, [2, 1, 2, 1, 2], [128, 256, 256, 728, 728]),
            "middle_flow": (16, 1, 728),
            "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])
        }
    else:
        raise ValueError(
            "Xception backbont only support xception_41/xception_65/xception_71")
    return bottleneck_params


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride=1,
                 padding=0,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
        self._bn = layers.SyncBatchNorm(
            num_features=output_channels, epsilon=1e-3, momentum=0.99)

        self._act_op = layers.Activation(act=act)

    def forward(self, inputs):
        return self._act_op(self._bn(self._conv(inputs)))


class Seperate_Conv(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride,
                 filter,
                 dilation=1,
                 act=None,
                 name=None):
        super(Seperate_Conv, self).__init__()

        self._conv1 = nn.Conv2D(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=filter,
            stride=stride,
            groups=input_channels,
            padding=(filter) // 2 * dilation,
            dilation=dilation,
            bias_attr=False)
        self._bn1 = layers.SyncBatchNorm(
            input_channels, epsilon=1e-3, momentum=0.99)

        self._act_op1 = layers.Activation(act=act)

        self._conv2 = nn.Conv2D(
            input_channels,
            output_channels,
            1,
            stride=1,
            groups=1,
            padding=0,
            bias_attr=False)
        self._bn2 = layers.SyncBatchNorm(
            output_channels, epsilon=1e-3, momentum=0.99)

        self._act_op2 = layers.Activation(act=act)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._bn1(x)
        x = self._act_op1(x)
        x = self._conv2(x)
        x = self._bn2(x)
        x = self._act_op2(x)
        return x


class Xception_Block(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 strides=1,
                 filter_size=3,
                 dilation=1,
                 skip_conv=True,
                 has_skip=True,
                 activation_fn_in_separable_conv=False,
                 name=None):
        super(Xception_Block, self).__init__()

        repeat_number = 3
        output_channels = check_data(output_channels, repeat_number)
        filter_size = check_data(filter_size, repeat_number)
        strides = check_data(strides, repeat_number)

        self.has_skip = has_skip
        self.skip_conv = skip_conv
        self.activation_fn_in_separable_conv = activation_fn_in_separable_conv
        if not activation_fn_in_separable_conv:
            self._conv1 = Seperate_Conv(
                input_channels,
                output_channels[0],
                stride=strides[0],
                filter=filter_size[0],
                dilation=dilation,
                name=name + "/separable_conv1")
            self._conv2 = Seperate_Conv(
                output_channels[0],
                output_channels[1],
                stride=strides[1],
                filter=filter_size[1],
                dilation=dilation,
                name=name + "/separable_conv2")
            self._conv3 = Seperate_Conv(
                output_channels[1],
                output_channels[2],
                stride=strides[2],
                filter=filter_size[2],
                dilation=dilation,
                name=name + "/separable_conv3")
        else:
            self._conv1 = Seperate_Conv(
                input_channels,
                output_channels[0],
                stride=strides[0],
                filter=filter_size[0],
                act="relu",
                dilation=dilation,
                name=name + "/separable_conv1")
            self._conv2 = Seperate_Conv(
                output_channels[0],
                output_channels[1],
                stride=strides[1],
                filter=filter_size[1],
                act="relu",
                dilation=dilation,
                name=name + "/separable_conv2")
            self._conv3 = Seperate_Conv(
                output_channels[1],
                output_channels[2],
                stride=strides[2],
                filter=filter_size[2],
                act="relu",
                dilation=dilation,
                name=name + "/separable_conv3")

        if has_skip and skip_conv:
            self._short = ConvBNLayer(
                input_channels,
                output_channels[-1],
                1,
                stride=strides[-1],
                padding=0,
                name=name + "/shortcut")

    def forward(self, inputs):
        if not self.activation_fn_in_separable_conv:
            x = F.relu(inputs)
            x = self._conv1(x)
            x = F.relu(x)
            x = self._conv2(x)
            x = F.relu(x)
            x = self._conv3(x)
        else:
            x = self._conv1(inputs)
            x = self._conv2(x)
            x = self._conv3(x)
        if self.has_skip is False:
            return x
        if self.skip_conv:
            skip = self._short(inputs)
        else:
            skip = inputs
        return x + skip


class XceptionDeeplab(nn.Layer):
    """
    The Xception backobne of DeepLabv3+ implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

     Args:
         backbone (str): Which type of Xception_DeepLab to select. It should be one of ('xception_41', 'xception_65', 'xception_71').
         pretrained (str, optional): The path of pretrained model.
         output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 16.

    """

    def __init__(self, backbone, pretrained=None, output_stride=16):

        super(XceptionDeeplab, self).__init__()

        bottleneck_params = gen_bottleneck_params(backbone)
        self.backbone = backbone
        self.feat_channels = [128, 2048]

        self._conv1 = ConvBNLayer(
            3,
            32,
            3,
            stride=2,
            padding=1,
            act="relu",
            name=self.backbone + "/entry_flow/conv1")
        self._conv2 = ConvBNLayer(
            32,
            64,
            3,
            stride=1,
            padding=1,
            act="relu",
            name=self.backbone + "/entry_flow/conv2")
        """
            bottleneck_params = {
            "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
            "middle_flow": (16, 1, 728),
            "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])
        }

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)

        """
        self.block_num = bottleneck_params["entry_flow"][0]
        self.strides = bottleneck_params["entry_flow"][1]
        self.chns = bottleneck_params["entry_flow"][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)

        self.entry_flow = []
        self.middle_flow = []

        self.stride = 2
        self.output_stride = output_stride
        s = self.stride

        for i in range(self.block_num):
            stride = self.strides[i] if check_stride(s * self.strides[i],
                                                     self.output_stride) else 1
            xception_block = self.add_sublayer(
                self.backbone + "/entry_flow/block" + str(i + 1),
                Xception_Block(
                    input_channels=64 if i == 0 else self.chns[i - 1],
                    output_channels=self.chns[i],
                    strides=[1, 1, self.stride],
                    name=self.backbone + "/entry_flow/block" + str(i + 1)))
            self.entry_flow.append(xception_block)
            s = s * stride
        self.stride = s

        self.block_num = bottleneck_params["middle_flow"][0]
        self.strides = bottleneck_params["middle_flow"][1]
        self.chns = bottleneck_params["middle_flow"][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)
        s = self.stride

        for i in range(self.block_num):
            stride = self.strides[i] if check_stride(s * self.strides[i],
                                                     self.output_stride) else 1
            xception_block = self.add_sublayer(
                self.backbone + "/middle_flow/block" + str(i + 1),
                Xception_Block(
                    input_channels=728,
                    output_channels=728,
                    strides=[1, 1, self.strides[i]],
                    skip_conv=False,
                    name=self.backbone + "/middle_flow/block" + str(i + 1)))
            self.middle_flow.append(xception_block)
            s = s * stride
        self.stride = s

        self.block_num = bottleneck_params["exit_flow"][0]
        self.strides = bottleneck_params["exit_flow"][1]
        self.chns = bottleneck_params["exit_flow"][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)
        s = self.stride
        stride = self.strides[0] if check_stride(s * self.strides[0],
                                                 self.output_stride) else 1
        self._exit_flow_1 = Xception_Block(
            728,
            self.chns[0], [1, 1, stride],
            name=self.backbone + "/exit_flow/block1")
        s = s * stride
        stride = self.strides[1] if check_stride(s * self.strides[1],
                                                 self.output_stride) else 1
        self._exit_flow_2 = Xception_Block(
            self.chns[0][-1],
            self.chns[1], [1, 1, stride],
            dilation=2,
            has_skip=False,
            activation_fn_in_separable_conv=True,
            name=self.backbone + "/exit_flow/block2")

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        feat_list = []
        for i, ef in enumerate(self.entry_flow):
            x = ef(x)
            if i == 0:
                feat_list.append(x)
        for mf in self.middle_flow:
            x = mf(x)
        x = self._exit_flow_1(x)
        x = self._exit_flow_2(x)
        feat_list.append(x)
        return feat_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
def Xception41_deeplab(**args):
    model = XceptionDeeplab('xception_41', **args)
    return model


@manager.BACKBONES.add_component
def Xception65_deeplab(**args):
    model = XceptionDeeplab("xception_65", **args)
    return model


@manager.BACKBONES.add_component
def Xception71_deeplab(**args):
    model = XceptionDeeplab("xception_71", **args)
    return model
