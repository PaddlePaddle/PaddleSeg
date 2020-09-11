# -*- encoding: utf-8 -*-
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

import paddle
import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import Conv2D
#from paddle.nn import SyncBatchNorm as BatchNorm
from paddle.fluid.dygraph import SyncBatchNorm as BatchNorm

from dygraph.models.architectures import layer_utils


class FCNHead(fluid.dygraph.Layer):
    """
    The FCNHead implementation used in auxilary layer

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // 4
        self.conv_bn_relu = layer_utils.ConvBnRelu(num_channels=in_channels,
                                                   num_filters=inter_channels,
                                                   filter_size=3,
                                                   padding=1)

        self.conv = Conv2D(num_channels=inter_channels,
                           num_filters=out_channels,
                           filter_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = F.dropout(x, dropout_prob=0.1)
        x = self.conv(x)
        return x

class AuxLayer(fluid.dygraph.Layer):
    """
    The auxilary layer implementation for auxilary loss

    Args:
        in_channels (int): the number of input channels.
        inter_channels (int): intermediate channels.
        out_channels (int): the number of output channels, which is usually num_classes.
    """

    def __init__(self, in_channels, inter_channels, out_channels):
        super(AuxLayer, self).__init__()

        self.conv_bn_relu = layer_utils.ConvBnRelu(num_channels=in_channels,
                                                   num_filters=inter_channels,
                                                   filter_size=3,
                                                   padding=1)

        self.conv = Conv2D(num_channels=inter_channels,
                           num_filters=out_channels,
                           filter_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = F.dropout(x, dropout_prob=0.1)
        x = self.conv(x)
        return x

def get_loss(logit, label, ignore_index=255, EPS=1e-5):
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
    mask = label != ignore_index
    mask = fluid.layers.cast(mask, 'float32')
    loss, probs = fluid.layers.softmax_with_cross_entropy(
        logit,
        label,
        ignore_index=ignore_index,
        return_softmax=True,
        axis=-1)

    loss = loss * mask
    avg_loss = paddle.mean(loss) / (paddle.mean(mask) + EPS)

    label.stop_gradient = True
    mask.stop_gradient = True

    return avg_loss


def get_pred_score_map(logit):
    """
    Get prediction and score map output in inference phase.

    Args:
        logit (tensor): output logit of network

    Returns:
        pred (tensor): predition map
        score_map (tensor): score map
    """
    score_map = F.softmax(logit, axis=1)
    score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
    pred = fluid.layers.argmax(score_map, axis=3)
    pred = fluid.layers.unsqueeze(pred, axes=[3])

    return pred, score_map