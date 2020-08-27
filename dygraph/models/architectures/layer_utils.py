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

import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import Conv2D
from paddle.nn import SyncBatchNorm as BatchNorm
from paddle.nn.layer import activation


class ConvBnRelu(dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 using_sep_conv=False,
                 **kwargs):

        super(ConvBnRelu, self).__init__()

        if using_sep_conv:
            self.conv = DepthwiseConvBnRelu(num_channels,
                                            num_filters,
                                            filter_size,
                                            **kwargs)
        else:

            self.conv = Conv2D(num_channels,
                               num_filters,
                               filter_size,
                               **kwargs)

        self.batch_norm = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class ConvBn(dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = Conv2D(num_channels,
                           num_filters,
                           filter_size,
                           **kwargs)
        self.batch_norm = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class ConvReluPool(dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(ConvReluPool, self).__init__()
        self.conv = Conv2D(num_channels,
                           num_filters,
                           filter_size=3,
                           stride=1,
                           padding=1,
                           dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class ConvBnReluUpsample(dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(ConvBnReluUpsample, self).__init__()
        self.conv_bn_relu = ConvBnRelu(num_channels, num_filters)

    def forward(self, x, upsample_scale=2):
        x = self.conv_bn_relu(x)
        new_shape = [x.shape[2] * upsample_scale, x.shape[3] * upsample_scale]
        x = fluid.layers.resize_bilinear(x, new_shape)
        return x


class DepthwiseConvBnRelu(dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 **kwargs):
        super(DepthwiseConvBnRelu, self).__init__()
        self.depthwise_conv = ConvBn(num_channels,
                                     num_filters=num_channels,
                                     filter_size=filter_size,
                                     groups=num_channels,
                                     use_cudnn=False,
                                     **kwargs)
        self.piontwise_conv = ConvBnRelu(num_channels,
                                         num_filters,
                                         filter_size=1,
                                         groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class Activation(fluid.dygraph.Layer):
    """
    The wrapper of activations
    For example:
        >>> relu = Activation("relu")
        >>> print(relu)
        <class 'paddle.nn.layer.activation.ReLU'>
        >>> sigmoid = Activation("sigmoid")
        >>> print(sigmoid)
        <class 'paddle.nn.layer.activation.Sigmoid'>
        >>> not_exit_one = Activation("not_exit_one")
        KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink', 
        'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax', 
        'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"

    Args:
        act (str): the activation name in lowercase
    """

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = activation.__all__
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(act, act_dict.keys()))

    def forward(self, x):

        if self._act is not None:
            return self.act_func(x)
        else:
            return x