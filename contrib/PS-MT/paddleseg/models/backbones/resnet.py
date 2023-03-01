import paddle
from paddle import Tensor
import paddle.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'wide_resnet101_2'
]


class BasicBlock(nn.Layer):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Layer] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Layer]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2D(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion: int = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 norm_layer=None,
                 bn_eps=1e-5,
                 bn_momentum=0.9,
                 downsample=None,
                 dilation: int = 1,
                 padding: int = 1,
                 ):
        super(Bottleneck, self).__init__()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum, use_global_stats=True)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, bias_attr=False, dilation=dilation,
                               padding=padding)
        self.bn2 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum, use_global_stats=True)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion, epsilon=bn_eps, momentum=bn_momentum, use_global_stats=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 norm_layer=nn.BatchNorm2D,
                 in_channels=3,
                 bn_eps=1e-05,
                 bn_momentum=0.9, deep_stem=False, stem_width=32, pretrained=None):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2D(in_channels, stem_width, kernel_size=3, stride=2, padding=1,
                          bias_attr=False),
                norm_layer(stem_width, epsilon=bn_eps, momentum=bn_momentum, use_global_stats=True),
                nn.ReLU(),
                nn.Conv2D(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias_attr=False),
                norm_layer(stem_width, epsilon=bn_eps, momentum=bn_momentum, use_global_stats=True),
                nn.ReLU(),
                nn.Conv2D(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias_attr=False)
            )
        else:
            self.conv1 = nn.Conv2D(in_channels, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, epsilon=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

        self.layer4 = self._make_layer_4(block, norm_layer, 512, layers[3],
                                         stride=1,
                                         bn_eps=bn_eps, bn_momentum=bn_momentum,
                                         dilation=2, padding=2)
        self.pretrained = pretrained
        self.init_weight()

    def _make_layer(self, block, norm_layer, planes, blocks,
                    stride=1, bn_eps=1e-5, bn_momentum=0.9):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion, epsilon=bn_eps, momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer, bn_momentum=bn_momentum,
                    bn_eps=bn_eps))

        return nn.Sequential(*layers)

    # 第四层进行空洞卷积
    def _make_layer_4(self, block, norm_layer, planes, blocks,
                      stride=1, bn_eps=1e-5, bn_momentum=0.9, dilation=1, padding=0):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                norm_layer(planes * block.expansion, epsilon=bn_eps, momentum=bn_momentum),
            )

        layers = [block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, dilation, padding)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            dilation *= 2
            padding = dilation
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer, bn_momentum=bn_momentum,
                    bn_eps=bn_eps, dilation=dilation, padding=padding))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> List[Any]:
        # breakpoint()
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x);
        blocks.append(x)
        x = self.layer2(x);
        blocks.append(x)
        x = self.layer3(x);
        blocks.append(x)
        x = self.layer4(x);
        blocks.append(x)
        # breakpoint()
        return blocks

    def forward(self, x: Tensor) -> List[Any]:
        return self._forward_impl(x)

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)


def _resnet(arch: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


@manager.BACKBONES.add_component
def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


@manager.BACKBONES.add_component
def resnet101(**kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(**kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)
