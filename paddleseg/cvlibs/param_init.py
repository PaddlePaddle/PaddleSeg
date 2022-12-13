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

import paddle.nn as nn
import math


def uniform_init(param, **kwargs):
    """
    Initialize the `param` with uniform distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 2)
        param_init.uniform_init(linear.bias,  low=-0.5, high=0。5)
        print(linear.bias.numpy())
        # result is [-0.2734719   0.23939109]

    """
    initializer = nn.initializer.Uniform(**kwargs)
    initializer(param, param.block)


def constant_init(param, **kwargs):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.constant_init(linear.weight, value=2.0)
        print(linear.weight.numpy())
        # result is [[2. 2. 2. 2.], [2. 2. 2. 2.]]

    """
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    """
    Initialize the `param` with a Normal distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.normal_init(linear.weight, loc=0.0, scale=1.0)

    """
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with Kaiming Normal initialization.

    This function implements the `param` initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}
    In case of Normal distribution, the mean is 0 and the standard deviation
    is
    .. math::
        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        # uniform is used to decide whether to use uniform or normal distribution
        param_init.kaiming_normal_init(linear.weight)

    """
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def trunc_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with The Random TruncatedNormal (Gaussian) distribution initializer.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.trunc_normal_init(linear.weight, mean=0.0, std=0.02)

    """
    initializer = nn.initializer.TruncatedNormal(**kwargs)
    initializer(param, param.block)


def kaiming_uniform(param, **kwargs):
    r"""Implements the Kaiming Uniform initializer
    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.kaiming_uniform(linear.weight)
    """

    initializer = nn.initializer.KaimingUniform(**kwargs)
    initializer(param, param.block)


def xavier_uniform(param, **kwargs):
    r"""
    This implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.
    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where
    .. math::
        x = \sqrt{\frac{6.0}{fan\_in + fan\_out}}
    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.xavier_uniform(linear.weight)
    """
    initializer = nn.initializer.XavierUniform(**kwargs)
    initializer(param, param.block)


def c2_xavier_fill(layer):
    try:
        kaiming_uniform(
            layer.weight, negative_slope=1, nonlinearity='leaky_relu')
    except AttributeError:
        pass
    try:
        if layer.bias is not None:
            constant_init(layer.bias, value=0)
    except AttributeError:
        pass


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def c2_linear_fill(layer):
    import math
    kaiming_uniform(
        layer.weight, negative_slope=math.sqrt(5), nonlinearity='leaky_relu')
    if layer.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_init(layer.bias, low=-bound, high=bound)


def th_multihead_fill(layer, qkv_same_embed_dim=True):
    def _init_param_as_combined_linear_weight(p):
        bound = math.sqrt(6 / (3 * p.shape[0] + p.shape[1]))
        nn.initializer.Uniform(low=-bound, high=bound)(p)

    if qkv_same_embed_dim:
        _init_param_as_combined_linear_weight(layer.q_proj.weight)
        _init_param_as_combined_linear_weight(layer.k_proj.weight)
        _init_param_as_combined_linear_weight(layer.v_proj.weight)
        xavier_uniform(layer.out_proj.weight)
    else:
        for p in layer.parameters():
            if p.dim() > 1:
                xavier_uniform(p)


def th_linear_fill(layer):
    nn.initializer.KaimingUniform(
        negative_slope=math.sqrt(5), nonlinearity='leaky_relu')(layer.weight)
    if getattr(layer, 'bias', None) is not None:
        fan_in = layer.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.initializer.Uniform(low=-bound, high=bound)(layer.bias)


class THLinearInitMixin(object):
    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                th_linear_fill(layer)
