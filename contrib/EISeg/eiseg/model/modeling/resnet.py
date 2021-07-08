import paddle.nn as nn
from .resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s


class ResNetBackbone(nn.Layer):
    def __init__(self, backbone='resnet50', pretrained_base=True, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()

        if backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')

        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x, additional_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if additional_features is not None:
            x = x + nn.functional.pad(additional_features,
                                      [0, 0, 0, 0, 0, x.shape(1) - additional_features.shape(1)],
                                      mode='constant', value=0)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4

