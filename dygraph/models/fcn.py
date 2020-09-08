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
from dygraph.cvlibs import param_init

__all__ = [
    "fcn_hrnet_w18_small_v1", "fcn_hrnet_w18_small_v2", "fcn_hrnet_w18",
    "fcn_hrnet_w30", "fcn_hrnet_w32", "fcn_hrnet_w40", "fcn_hrnet_w44",
    "fcn_hrnet_w48", "fcn_hrnet_w60", "fcn_hrnet_w64"
]


@manager.MODELS.add_component
class FCN(fluid.dygraph.Layer):
    """
    Fully Convolutional Networks for Semantic Segmentation.
    https://arxiv.org/abs/1411.4038

    Args:
        num_classes (int): the unique number of target classes.

        backbone (paddle.nn.Layer): backbone networks.

        model_pretrained (str): the path of pretrained model.

        backbone_indices (tuple): one values in the tuple indicte the indices of output of backbone.Default -1.

        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.

        channels (int): channels after conv layer before the last one.

        ignore_index (int): the value of ground-truth mask would be ignored while computing loss or doing evaluation. Default 255.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 model_pretrained=None,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 ignore_index=255,
                 **kwargs):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        self.ignore_index = ignore_index
        self.EPS = 1e-5
        if channels is None:
            channels = backbone_channels[backbone_indices[0]]

        self.backbone = backbone
        self.conv_last_2 = ConvBNLayer(
            num_channels=backbone_channels[backbone_indices[0]],
            num_filters=channels,
            filter_size=1,
            stride=1)
        self.conv_last_1 = Conv2D(
            num_channels=channels,
            num_filters=self.num_classes,
            filter_size=1,
            stride=1,
            padding=0)
        self.init_weight(model_pretrained)

    def forward(self, x):
        input_shape = x.shape[2:]
        fea_list = self.backbone(x)
        x = fea_list[self.backbone_indices[0]]
        x = self.conv_last_2(x)
        logit = self.conv_last_1(x)
        logit = fluid.layers.resize_bilinear(logit, input_shape)
        return [logit]

        # if self.training:
        #     if label is None:
        #         raise Exception('Label is need during training')
        #     return self._get_loss(logit, label)
        # else:
        #     score_map = fluid.layers.softmax(logit, axis=1)
        #     score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
        #     pred = fluid.layers.argmax(score_map, axis=3)
        #     pred = fluid.layers.unsqueeze(pred, axes=[3])
        #     return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model. Defaults to None.
        """
        params = self.parameters()
        for param in params:
            param_name = param.name
            if 'batch_norm' in param_name:
                if 'w_0' in param_name:
                    param_init.constant_init(param, 1.0)
                elif 'b_0' in param_name:
                    param_init.constant_init(param, 0.0)
            if 'conv' in param_name and 'w_0' in param_name:
                param_init.normal_init(param, scale=0.001)

        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))

    # def _get_loss(self, logit, label):
    #     """
    #     compute forward loss of the model

    #     Args:
    #         logit (tensor): the logit of model output
    #         label (tensor): ground truth

    #     Returns:
    #         avg_loss (tensor): forward loss
    #     """
    #     logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    #     label = fluid.layers.transpose(label, [0, 2, 3, 1])
    #     mask = label != self.ignore_index
    #     mask = fluid.layers.cast(mask, 'float32')
    #     loss, probs = fluid.layers.softmax_with_cross_entropy(
    #         logit,
    #         label,
    #         ignore_index=self.ignore_index,
    #         return_softmax=True,
    #         axis=-1)

    #     loss = loss * mask
    #     avg_loss = fluid.layers.mean(loss) / (
    #         fluid.layers.mean(mask) + self.EPS)

    #     label.stop_gradient = True
    #     mask.stop_gradient = True
    #     return avg_loss


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu"):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        self._batch_norm = BatchNorm(num_filters)
        self.act = act

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        if self.act == 'relu':
            y = fluid.layers.relu(y)
        return y


@manager.MODELS.add_component
def fcn_hrnet_w18_small_v1(*args, **kwargs):
    return FCN(backbone='HRNet_W18_Small_V1', backbone_channels=(240), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w18_small_v2(*args, **kwargs):
    return FCN(backbone='HRNet_W18_Small_V2', backbone_channels=(270), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w18(*args, **kwargs):
    return FCN(backbone='HRNet_W18', backbone_channels=(270), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w30(*args, **kwargs):
    return FCN(backbone='HRNet_W30', backbone_channels=(450), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w32(*args, **kwargs):
    return FCN(backbone='HRNet_W32', backbone_channels=(480), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w40(*args, **kwargs):
    return FCN(backbone='HRNet_W40', backbone_channels=(600), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w44(*args, **kwargs):
    return FCN(backbone='HRNet_W44', backbone_channels=(660), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w48(*args, **kwargs):
    return FCN(backbone='HRNet_W48', backbone_channels=(720), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w60(*args, **kwargs):
    return FCN(backbone='HRNet_W60', backbone_channels=(900), **kwargs)


@manager.MODELS.add_component
def fcn_hrnet_w64(*args, **kwargs):
    return FCN(backbone='HRNet_W64', backbone_channels=(960), **kwargs)
