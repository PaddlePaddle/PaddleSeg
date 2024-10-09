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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
import paddle.distributed as dist
import math


class _AllReduce(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            paddle.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, sync_op=True)
        inputs = paddle.stack(input_list, axis=0)
        return paddle.sum(inputs, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, sync_op=True)
        return grad_output


def differentiable_all_reduce(input):
    """
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (not dist.is_available() or not dist.is_initialized()
            or dist.get_world_size() == 1):
        return input
    return _AllReduce.apply(input)


class NaiveSyncBatchNorm(nn.BatchNorm2D):

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if dist.get_world_size() == 1 or not self.training:
            return super(NaiveSyncBatchNorm, self).forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = paddle.mean(input, axis=[0, 2, 3])
        meansqr = paddle.mean(input * input, axis=[0, 2, 3])

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = paddle.concat([mean, meansqr], axis=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = paddle.split(vec, [C, C])
            momentum = 1 - self._momentum  # NOTE: paddle has reverse momentum defination
        else:
            if B == 0:
                vec = paddle.zeros([2 * C + 1], dtype=mean.dtype)
                vec = vec + input.sum(
                )  # make sure there is gradient w.r.t input
            else:
                vec = paddle.concat(
                    [
                        mean,
                        meansqr,
                        paddle.ones([1], dtype=mean.dtype),
                    ],
                    axis=0,
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clip(max=1) * (
                1 - self._momentum)  # no update if total_batch is 0
            mean, meansqr, _ = paddle.split(
                vec / total_batch.clip(min=1),
                [C, C, int(vec.shape[0] - 2 * C)])  # avoid div-by-zero

        var = meansqr - mean * mean
        invstd = paddle.rsqrt(var + self._epsilon)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape([1, -1, 1, 1])
        bias = bias.reshape([1, -1, 1, 1])

        tmp_mean = self._mean + momentum * (mean.detach() - self._mean)
        self._mean.set_value(tmp_mean)
        tmp_variance = self._variance + (momentum *
                                         (var.detach() - self._variance))
        self._variance.set_value(tmp_variance)
        ret = input * scale + bias
        return ret

    @classmethod
    def convert_sync_batchnorm(cls, layer):
        layer_output = layer
        if isinstance(layer, nn.BatchNorm2D):

            layer_output = NaiveSyncBatchNorm(layer._num_features,
                                              layer._momentum, layer._epsilon,
                                              layer._weight_attr,
                                              layer._bias_attr,
                                              layer._data_format, layer._name)

            if (layer._weight_attr is not False
                    and layer._bias_attr is not False):
                with paddle.no_grad():
                    layer_output.weight = layer.weight
                    layer_output.bias = layer.bias
            layer_output._mean = layer._mean
            layer_output._variance = layer._variance

        for name, sublayer in layer.named_children():
            layer_output.add_sublayer(name,
                                      cls.convert_sync_batchnorm(sublayer))
        del layer
        return layer_output


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get(
            'PADDLESEG_EXPORT_STAGE') or 'xpu' in paddle.get_device():
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    elif 'npu' in paddle.get_device() or 'mlu' in paddle.get_device():
        return NaiveSyncBatchNorm(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = layers.Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvGNAct(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding="same",
                 num_groups=32,
                 act_type=None,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if "data_format" in kwargs:
            data_format = kwargs["data_format"]
        else:
            data_format = "NCHW"
        self._group_norm = nn.GroupNorm(num_groups,
                                        out_channels,
                                        data_format=data_format)
        self._act_type = act_type
        if act_type is not None:
            self._act = layers.Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        x = self._group_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvNormAct(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 norm=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'

        self._norm = norm if norm is not None else None

        self._act_type = act_type
        if act_type is not None:
            self._act = layers.Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        if self._norm is not None:
            x = self._norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBNAct(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

        self._act_type = act_type
        if act_type is not None:
            self._act = layers.Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBN(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1)
        self._relu = layers.Activation("relu")
        self._max_pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(in_channels,
                                     out_channels=in_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     groups=in_channels,
                                     **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         groups=1,
                                         data_format=data_format,
                                         bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     groups=in_channels,
                                     **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(in_channels=in_channels,
                                       out_channels=inter_channels,
                                       kernel_size=3,
                                       padding=1,
                                       **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(in_channels=inter_channels,
                              out_channels=out_channels,
                              kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class JPU(nn.Layer):
    """
    Joint Pyramid Upsampling of FCN.
    The original paper refers to
        Wu, Huikai, et al. "Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation." arXiv preprint arXiv:1903.11816 (2019).
    """

    def __init__(self, in_channels, width=512):
        super().__init__()

        self.conv5 = ConvBNReLU(in_channels[-1],
                                width,
                                3,
                                padding=1,
                                bias_attr=False)
        self.conv4 = ConvBNReLU(in_channels[-2],
                                width,
                                3,
                                padding=1,
                                bias_attr=False)
        self.conv3 = ConvBNReLU(in_channels[-3],
                                width,
                                3,
                                padding=1,
                                bias_attr=False)

        self.dilation1 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=1,
            pointwise_bias=False,
            dilation=1,
            bias_attr=False,
            stride=1,
        )
        self.dilation2 = SeparableConvBNReLU(3 * width,
                                             width,
                                             3,
                                             padding=2,
                                             pointwise_bias=False,
                                             dilation=2,
                                             bias_attr=False,
                                             stride=1)
        self.dilation3 = SeparableConvBNReLU(3 * width,
                                             width,
                                             3,
                                             padding=4,
                                             pointwise_bias=False,
                                             dilation=4,
                                             bias_attr=False,
                                             stride=1)
        self.dilation4 = SeparableConvBNReLU(3 * width,
                                             width,
                                             3,
                                             padding=8,
                                             pointwise_bias=False,
                                             dilation=8,
                                             bias_attr=False,
                                             stride=1)

    def forward(self, *inputs):
        feats = [
            self.conv5(inputs[-1]),
            self.conv4(inputs[-2]),
            self.conv3(inputs[-3])
        ]
        size = feats[-1].shape[2:]
        feats[-2] = F.interpolate(feats[-2],
                                  size,
                                  mode='bilinear',
                                  align_corners=True)
        feats[-3] = F.interpolate(feats[-3],
                                  size,
                                  mode='bilinear',
                                  align_corners=True)

        feat = paddle.concat(feats, axis=1)
        feat = paddle.concat([
            self.dilation1(feat),
            self.dilation2(feat),
            self.dilation3(feat),
            self.dilation4(feat)
        ],
                             axis=1)

        return inputs[0], inputs[1], inputs[2], feat


class ConvBNPReLU(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._prelu = layers.Activation("prelu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x


class ConvBNLeakyReLU(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = layers.Activation("leakyrelu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x
