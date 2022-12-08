# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = [
    'LPSNet',
    'lps_net_small',
    'lps_net_medium',
    'lps_net_large',
]

_interpolate = partial(F.interpolate, mode="bilinear", align_corners=True)


@manager.MODELS.add_component
class LPSNet(nn.Layer):
    DEPTHS = 0
    WIDTHS = 1
    RESOLUTIONS = 2
    """
    The LPSNet implementation based on PaddlePaddle.

    The original article refers to
    Zhang, Yiheng and Yao, Ting and Qiu, Zhaofan and Mei, Tao. "Lightweight and Progressively-Scalable Networks for Semantic Segmentation"
    (https://arxiv.org/pdf/2207.13600)

    Args:
        depths (list): Depths of each block.
        channels (lsit): Channels of each block.
        scale_ratios (list): Scale ratio for each branch. The number of branchs depends on length of scale_ratios.
        num_classes (int): The unique number of target classes.
        deploy (bool): Whether use reparameterization. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
            self,
            depths,
            channels,
            scale_ratios,
            num_classes,
            deploy=False,
            pretrained=None, ):
        super().__init__()

        self.deploy = deploy
        self.depths = depths
        self.channels = channels
        self.scale_ratios = list(filter(lambda x: x > 0, scale_ratios))
        self.num_classes = num_classes

        self.num_paths = len(self.scale_ratios)
        self.num_blocks = len(depths)

        if self.num_blocks != len(self.channels):
            raise ValueError(
                f"Expect depths and channels have save length, but got {self.num_blocks} and {len(self.channels)}"
            )

        self.nets = nn.LayerList(
            [self._build_path() for _ in range(self.num_paths)])

        self.head = nn.Conv2D(
            channels[-1] * self.num_paths, num_classes, 1, bias_attr=True)

        self._init_weight(pretrained)

    def _init_weight(self, pretrained):
        if pretrained is not None:
            utils.load_entire_model(self, pretrained)

    def _build_path(self):
        path = []
        c_in = 3
        for b, (d, c) in enumerate(zip(self.depths, self.channels)):
            blocks = []
            for i in range(d):
                blocks.append(
                    ConvBNReLU(
                        in_channels=c_in if i == 0 else c,
                        out_channels=c,
                        kernel_size=3,
                        padding=1,
                        deploy=self.deploy,
                        stride=2
                        if (i == 0 and b != self.num_blocks - 1) else 1, ))
                c_in = c
            path.append(nn.Sequential(*blocks))
        return nn.LayerList(path)

    def switch_to_deploy(self):
        if self.deploy:
            return

        for _, layer in self.named_sublayers():
            if isinstance(layer, ConvBNReLU):
                layer.switch_to_deploy()

        self.eval()

    @classmethod
    def expand(cls, module, delta, expand_type, deploy=False):
        if isinstance(expand_type, int):
            expand_type = [expand_type]
            delta = [delta]

        depths = module.depths
        channels = module.channels
        scale_ratios = module.scale_ratios

        attrs = [depths, channels, scale_ratios]
        for d, e in zip(delta, expand_type):
            if len(d) != len(attrs[e]):
                raise ValueError(
                    f"Expect elements of delta and original attribution have save length, but got {len(d)} and {len(attrs[e])}"
                )
            attrs[e] = [a + b for a, b in zip(attrs[e], d)]

        model = cls(*attrs, num_classes=module.num_classes, deploy=deploy)
        return model

    def _preprocess_input(self, x):
        h, w = x.shape[-2:]
        return [
            _interpolate(x, (int(r * h), int(r * w))) for r in self.scale_ratios
        ]

    def forward(self, x, interact_begin_idx=2):
        input_size = paddle.shape(x)[-2:]
        inputs = self._preprocess_input(x)
        feats = []
        for path, x in zip(self.nets, inputs):
            inp = x
            for idx in range(interact_begin_idx + 1):
                inp = path[idx](inp)
            feats.append(inp)

        for idx in range(interact_begin_idx + 1, self.num_blocks):
            feats = _multipath_interaction(feats)
            feats = [path[idx](x) for path, x in zip(self.nets, feats)]

        size = feats[0].shape[-2:]
        feats = [_interpolate(x, size=size) for x in feats]

        out = self.head(paddle.concat(feats, 1))

        return [_interpolate(out, size=input_size)]


def _multipath_interaction(feats):
    length = len(feats)
    if length == 1:
        return feats[0]
    sizes = [x.shape[-2:] for x in feats]
    outs = []
    looper = list(range(length))
    for i, s in enumerate(sizes):
        out = feats[i]
        for j in filter(lambda x: x != i, looper):
            out += _interpolate(feats[j], size=s)
        outs.append(out)
    return outs


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 deploy=False,
                 **kwargs):
        super().__init__()
        if deploy:
            self.conv = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    **kwargs),
                nn.ReLU(), )
        else:
            kwargs['bias_attr'] = False
            self.conv = layers.ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size,
                padding,
                **kwargs, )

    def forward(self, x):
        return self.conv(x)

    def switch_to_deploy(self):
        # NOTE: works when use one gpu
        module = self.conv
        kernel = module._conv.weight
        conv_bias = module._conv.bias if module._conv.bias is not None else 0
        running_mean = module._batch_norm._mean
        running_var = module._batch_norm._variance
        gamma = module._batch_norm.weight
        beta = module._batch_norm.bias
        eps = module._batch_norm._epsilon

        std = paddle.sqrt(running_var + eps)
        t = (gamma / std).reshape((-1, 1, 1, 1))

        weight = kernel * t
        bias = beta + (conv_bias - running_mean) * gamma / std

        conv = nn.Conv2D(
            in_channels=module._conv._in_channels,
            out_channels=module._conv._out_channels,
            kernel_size=3,
            stride=module._conv._stride,
            padding=module._conv._padding)
        conv.weight.set_value(weight)
        conv.bias.set_value(bias)
        delattr(self, 'conv')
        self.conv = nn.Sequential(
            conv,
            nn.ReLU(), )


@manager.MODELS.add_component
def lps_net_small(deploy=False, num_classes=19, pretrained=None):
    depths = [1, 3, 3, 10, 10]
    channels = [8, 24, 48, 96, 96]
    scale_ratios = [3 / 4, 1 / 4]
    return LPSNet(depths, channels, scale_ratios, num_classes, deploy,
                  pretrained)


@manager.MODELS.add_component
def lps_net_medium(deploy=False, num_classes=19, pretrained=None):
    depths = [1, 3, 3, 10, 10]
    channels = [8, 24, 48, 96, 96]
    scale_ratios = [1, 1 / 4]
    return LPSNet(depths, channels, scale_ratios, num_classes, deploy,
                  pretrained)


@manager.MODELS.add_component
def lps_net_large(deploy=False, num_classes=19, pretrained=None):
    depths = [1, 3, 3, 10, 10]
    channels = [8, 24, 64, 128, 128]
    scale_ratios = [1, 1 / 4, 0]
    return LPSNet(depths, channels, scale_ratios, num_classes, deploy,
                  pretrained)
