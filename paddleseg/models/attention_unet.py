import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg import utils
import numpy as np


class AttentionBlock(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2D(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UpConv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Layer):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(input_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = filters
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel, channel * 2)
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class ConvBlock(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


@manager.MODELS.add_component
class AttentionUNet(nn.Layer):
    def __init__(self, n_channels=3, num_classes=1, pretrained=None):
        super().__init__()
        self.encoder = Encoder(n_channels, [64, 128, 256, 512])
        filters = np.array([64, 128, 256, 512, 1024])
        self.up5 = UpConv(ch_in=filters[4], ch_out=filters[3])
        self.att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.up_conv5 = ConvBlock(ch_in=filters[4], ch_out=filters[3])

        self.up4 = UpConv(ch_in=filters[3], ch_out=filters[2])
        self.att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.up_conv4 = ConvBlock(ch_in=filters[3], ch_out=filters[2])

        self.up3 = UpConv(ch_in=filters[2], ch_out=filters[1])
        self.att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.up_conv3 = ConvBlock(ch_in=filters[2], ch_out=filters[1])

        self.up2 = UpConv(ch_in=filters[1], ch_out=filters[0])
        self.att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.up_conv2 = ConvBlock(ch_in=filters[1], ch_out=filters[0])

        self.conv_1x1 = nn.Conv2D(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x5, (x1, x2, x3, x4) = self.encoder(x)
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = paddle.concat([x4, d5], axis=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = paddle.concat((x3, d4), axis=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = paddle.concat((x2, d3), axis=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = paddle.concat((x1, d2), axis=1)
        d2 = self.up_conv2(d2)

        logit = self.conv_1x1(d2)
        logit_list = []
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
