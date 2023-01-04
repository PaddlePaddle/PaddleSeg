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

# Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
# 
# Original copyright info: 

# Copyright (c) Facebook, Inc. and its affiliates.

# NOTE: This file implements vanilla ResNet architectures, which differ 
# from the ResNet-vd implementation in PaddleSeg.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.utils import utils

from paddlepanseg.cvlibs import manager
from ..common import Conv2D
from ..param_init import c2_msra_fill

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

DEFAULT_CONFIG = {
    'in_channels': 3,
    'stem_out_channels': 64,
    'res2_out_channels': 256,
    'stride_in_1x1': True,
    'num_groups': 1,
    'width_per_group': 64,
    'res5_dilation': 1,
}


class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias_attr=False,
                norm=nn.BatchNorm2D(out_channels), )
        else:
            self.shortcut = None

        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False,
            norm=nn.BatchNorm2D(out_channels), )

        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            norm=nn.BatchNorm2D(out_channels), )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            stride_in_1x1=False,
            dilation=1, ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            stride_in_1x1 (bool): when stride>1, whether or not to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias_attr=False,
                norm=nn.BatchNorm2D(out_channels), )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2D(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias_attr=False,
            norm=nn.BatchNorm2D(bottleneck_channels), )

        self.conv2 = Conv2D(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias_attr=False,
            groups=num_groups,
            dilation=dilation,
            norm=nn.BatchNorm2D(bottleneck_channels), )

        self.conv3 = Conv2D(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias_attr=False,
            norm=nn.BatchNorm2D(out_channels), )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(nn.Layer):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 4
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False,
            norm=nn.BatchNorm2D(out_channels), )
        c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(nn.Layer):
    def __init__(self, stem, stages, pretrained=None):
        """
        Args:
            stem (nn.Module): a stem module.
            stages (list[list[paddle.nn.Layer]]): several (typically 4) stages,
                each contains multiple :class:`paddle.nn.Layer`.
        """
        super().__init__()
        self.stem = stem

        self.feat_channels = []

        self.stage_names, self.stages = [], []

        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_sublayer(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self.feat_channels.append(blocks[-1].out_channels)
        self.stage_names = tuple(
            self.stage_names)  # Make it static for scripting
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim(
        ) == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels,
                   **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of paddle.nn.Layer that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[paddle.nn.Layer]: a list of block module.

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )
        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}.")
                    newk = k[:-len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **curr_kwargs))
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[paddle.nn.Layer]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels,
                                out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs, ))
        return ret

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


def build_resnet(**cfg):
    for key, val in DEFAULT_CONFIG.items():
        cfg.setdefault(key, val)

    stem = BasicStem(
        in_channels=cfg['in_channels'], out_channels=cfg['stem_out_channels'])

    depth = cfg['depth']
    num_groups = cfg['num_groups']
    width_per_group = cfg['width_per_group']
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg['stem_out_channels']
    out_channels = cfg['res2_out_channels']
    stride_in_1x1 = cfg['stride_in_1x1']
    res5_dilation = cfg['res5_dilation']

    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(
        res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set res2_out_channels = 64 for R18/R34"
        assert res5_dilation == 1, "Must set res5_dilation = 1 for R18/R34"
        assert num_groups == 1, "Must set num_groups = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and
                                         dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block":
            [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, pretrained=cfg.get('pretrained', None))


@manager.BACKBONES.add_component
def ResNet18(**cfg):
    return build_resnet(depth=18, **cfg)


@manager.BACKBONES.add_component
def ResNet34(**cfg):
    return build_resnet(depth=34, **cfg)


@manager.BACKBONES.add_component
def ResNet50(**cfg):
    return build_resnet(depth=50, **cfg)


@manager.BACKBONES.add_component
def ResNet101(**cfg):
    return build_resnet(depth=101, **cfg)


@manager.BACKBONES.add_component
def ResNet152(**cfg):
    return build_resnet(depth=152, **cfg)
