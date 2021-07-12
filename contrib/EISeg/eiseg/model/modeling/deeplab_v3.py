from contextlib import ExitStack

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .basic_blocks import SeparableConv2d
from paddleseg.models.backbones import *
from .resnet import ResNetBackbone

class DeepLabV3Plus(nn.Layer):
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2D,
                 backbone_norm_layer=None,
                 ch=256,
                 project_dropout=0.5,
                 inference_mode=False,
                 **kwargs):
        super(DeepLabV3Plus, self).__init__()
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer

        self.backbone_name = backbone
        self.norm_layer = norm_layer
        self.backbone_norm_layer = backbone_norm_layer
        self.inference_mode = False
        self.ch = ch
        self.aspp_in_channels = 2048
        self.skip_project_in_channels = 256  # layer 1 out_channels

        self._kwargs = kwargs
        if backbone == 'resnet34':
            self.aspp_in_channels = 512
            self.skip_project_in_channels = 64

        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False,
                                       norm_layer=self.backbone_norm_layer, **kwargs)
        
        
        self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch,
                                 norm_layer=self.norm_layer)
        self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
        self.aspp = _ASPP(in_channels=self.aspp_in_channels,
                          atrous_rates=[12, 24, 36],
                          out_channels=ch,
                          project_dropout=project_dropout,
                          norm_layer=self.norm_layer)

        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True,
                                    norm_layer=self.backbone_norm_layer, **self._kwargs)

        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = pretrained.state_dict()

        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict)

        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, x, additional_features=None):
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(paddle.no_grad())

            c1, _, c3, c4 = self.backbone(x, additional_features)
            c1 = self.skip_project(c1)

            x = self.aspp(c4)
            x = F.interpolate(x, c1.shape[2:], mode='bilinear', align_corners=True)
            x = paddle.concat((x, c1), axis=1)
            x = self.head(x)

        return x,


class _SkipProject(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2D):
        super(_SkipProject, self).__init__()

        self.skip_project = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.skip_project(x)


class _DeepLabHead(nn.Layer):
    def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2D):
        super(_DeepLabHead, self).__init__()

        self.block = nn.Sequential(
            SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3,
                            dw_padding=1, norm_layer=norm_layer),
            SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3,
                            dw_padding=1, norm_layer=norm_layer),
            nn.Conv2D(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Layer):
    def __init__(self, in_channels, atrous_rates, out_channels=256,
                 project_dropout=0.5, norm_layer=nn.BatchNorm2D):
        super(_ASPP, self).__init__()

        b0 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias_attr=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.concurent = nn.LayerList([b0, b1, b2, b3, b4])

        project = [
            nn.Conv2D(in_channels=5 * out_channels, out_channels=out_channels,
                      kernel_size=1, bias_attr=False),
            norm_layer(out_channels),
            nn.ReLU()
        ]
        if project_dropout > 0:
            project.append(nn.Dropout(project_dropout))
        self.project = nn.Sequential(*project)

    def forward(self, x):
        x = paddle.concat([block(x) for block in self.concurent], axis=1)

        return self.project(x)


class _AsppPooling(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, bias_attr=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, x.shape[2:], mode='bilinear', align_corners=True)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, padding=atrous_rate,
                  dilation=atrous_rate, bias_attr=False),
        norm_layer(out_channels),
        nn.ReLU()
    )

    return block
