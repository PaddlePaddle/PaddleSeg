import paddle
import paddle.nn as nn


class BasicBlockV1b(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNetV1b(nn.Layer):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm2d`)
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    def __init__(self, block, layers, classes=1000, dilated=True, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, norm_layer=nn.BatchNorm2D):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if not deep_stem:
            self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, stem_width, kernel_size=3, stride=2, padding=1, bias_attr=False),
                norm_layer(stem_width),
                nn.ReLU(),
                nn.Conv2D(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias_attr=False),
                norm_layer(stem_width),
                nn.ReLU(),
                nn.Conv2D(stem_width, 2*stem_width, kernel_size=3, stride=1, padding=1, bias_attr=False)
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down,
                                       norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,
                                           avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,
                                           avg_down=avg_down, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           avg_down=avg_down, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    avg_down=False, norm_layer=nn.BatchNorm2D):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = []
            if avg_down:
                if dilation == 1:
                    downsample.append(
                        nn.AvgPool2D(kernel_size=stride, stride=stride, ceil_mode=True)
                    )
                else:
                    downsample.append(
                        nn.AvgPool2D(kernel_size=1, stride=1, ceil_mode=True)
                    )
                downsample.extend([
                    nn.Conv2D(self.inplanes, out_channels=planes * block.expansion,
                              kernel_size=1, stride=1, bias_attr=False),
                    norm_layer(planes * block.expansion)
                ])
                downsample = nn.Sequential(*downsample)
            else:
                downsample = nn.Sequential(
                    nn.Conv2D(self.inplanes, out_channels=planes * block.expansion,
                              kernel_size=1, stride=stride, bias_attr=False),
                    norm_layer(planes * block.expansion)
                )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape([x.shape[0], -1])
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def _safe_state_dict_filtering(orig_dict, model_dict_keys):
    filtered_orig_dict = {}
    for k, v in orig_dict.items():
        if k in model_dict_keys:
            filtered_orig_dict[k] = v
        else:
            print(f"[ERROR] Failed to load <{k}> in backbone")
    return filtered_orig_dict


def resnet18_v1b(pretrained=False, pretrained_checkpoint=None,**kwargs):
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(
            paddle.load(pretrained_checkpoint),
            model_dict.keys()
        )
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34_v1b(pretrained=False, pretrained_checkpoint=None, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(
            paddle.load(pretrained_checkpoint),
            model_dict.keys()
        )
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_v1s(pretrained=False, pretrained_checkpoint=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(
            paddle.load(pretrained_checkpoint),
            model_dict.keys()
        )
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101_v1s(pretrained=False, pretrained_checkpoint=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(
            paddle.load(pretrained_checkpoint),
            model_dict.keys()
        )
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152_v1s(pretrained=False, pretrained_checkpoint=None, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(
            paddle.load(pretrained_checkpoint),
            model_dict.keys()
        )
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


