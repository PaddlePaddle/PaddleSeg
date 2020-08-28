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

import math
import os

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.initializer import Normal
from paddle.nn import SyncBatchNorm as BatchNorm

from dygraph.cvlibs import manager
from dygraph import utils

__all__ = [
    "fcn_hrnet_w18_small_v1", "fcn_hrnet_w18_small_v2", "fcn_hrnet_w18",
    "fcn_hrnet_w30", "fcn_hrnet_w32", "fcn_hrnet_w40", "fcn_hrnet_w44",
    "fcn_hrnet_w48", "fcn_hrnet_w60", "fcn_hrnet_w64"
]


class FCN(fluid.dygraph.Layer):
    """
    Fully Convolutional Networks for Semantic Segmentation.
    https://arxiv.org/abs/1411.4038

    Args:
        backbone (str): backbone name,
        num_classes (int): the unique number of target classes.
        in_channels (int): the channels of input feature maps.
        channels (int): channels after conv layer before the last one.
        pretrained_model (str): the path of pretrained model.
        ignore_index (int): the value of ground-truth mask would be ignored while computing loss or doing evaluation. Default 255.
    """

    def __init__(self,
                 backbone,
                 num_classes,
                 in_channels,
                 channels=None,
                 pretrained_model=None,
                 ignore_index=255,
                 **kwargs):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.EPS = 1e-5
        if channels is None:
            channels = in_channels

        self.backbone = manager.BACKBONES[backbone](**kwargs)
        self.conv_last_2 = ConvBNLayer(
            num_channels=in_channels,
            num_filters=channels,
            filter_size=1,
            stride=1,
            name='conv-2')
        self.conv_last_1 = Conv2D(
            num_channels=channels,
            num_filters=self.num_classes,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=ParamAttr(
                initializer=Normal(scale=0.001), name='conv-1_weights'))
        self.init_weight(pretrained_model)

    def forward(self, x, label=None, mode='train'):
        input_shape = x.shape[2:]
        x = self.backbone(x)
        x = self.conv_last_2(x)
        logit = self.conv_last_1(x)
        logit = fluid.layers.resize_bilinear(logit, input_shape)

        if self.training:
            if label is None:
                raise Exception('Label is need during training')
            return self._get_loss(logit, label)
        else:
            score_map = fluid.layers.softmax(logit, axis=1)
            score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
            pred = fluid.layers.argmax(score_map, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the pretrained_model path of backbone. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self.backbone, pretrained_model)
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))

    def _get_loss(self, logit, label):
        """
        compute forward loss of the model

        Args:
            logit (tensor): the logit of model output
            label (tensor): ground truth

        Returns:
            avg_loss (tensor): forward loss
        """
        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        label = fluid.layers.transpose(label, [0, 2, 3, 1])
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=-1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (
            fluid.layers.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            param_attr=ParamAttr(
                initializer=Normal(scale=0.001), name=name + "_weights"),
            bias_attr=False)
        bn_name = name + '_bn'
        self._batch_norm = BatchNorm(
            num_filters,
            weight_attr=ParamAttr(
                name=bn_name + '_scale',
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=ParamAttr(
                bn_name + '_offset',
                initializer=fluid.initializer.Constant(0.0)))
        self.act = act

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        if self.act == 'relu':
            y = fluid.layers.relu(y)
        return y


@manager.MODELS.add_component
def fcn_hrnet_w18_small_v1(*args, **kwargs):
    return FCN(backbone='HRNet_W18_Small_V1', in_channels=240, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w18_small_v2(*args, **kwargs):
    return FCN(backbone='HRNet_W18_Small_V2', in_channels=270, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w18(*args, **kwargs):
    return FCN(backbone='HRNet_W18', in_channels=270, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w30(*args, **kwargs):
    return FCN(backbone='HRNet_W30', in_channels=450, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w32(*args, **kwargs):
    return FCN(backbone='HRNet_W32', in_channels=480, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w40(*args, **kwargs):
    return FCN(backbone='HRNet_W40', in_channels=600, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w44(*args, **kwargs):
    return FCN(backbone='HRNet_W44', in_channels=660, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w48(*args, **kwargs):
    return FCN(backbone='HRNet_W48', in_channels=720, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w60(*args, **kwargs):
    return FCN(backbone='HRNet_W60', in_channels=900, **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w64(*args, **kwargs):
    return FCN(backbone='HRNet_W64', in_channels=960, **kwargs)
