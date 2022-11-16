# TODO add gpl license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddleseg
from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()


@manager.MODELS.add_component
class RVM(nn.Layer):
    """
    TODO add annotation
    """

    def __init__(
            self,
            backbone,
            lraspp_in_channels=960,
            lraspp_out_channels=128,
            pretrained=None, ):
        super().__init__()

        self.backbone = backbone
        self.aspp = LRASPP(lraspp_in_channels, lraspp_out_channels)

    def forward(self,
                src,
                r1=None,
                r2=None,
                r3=None,
                r4=None,
                downsample_ratio=1.,
                segmentation_pass=False):
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        f1, f2, f3, f4 = self.backbone(src)
        f4 = self.aspp(f4)
        print(f4.shape)


class LRASPP(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2D(
                in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(True))
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(
                in_channels, out_channels, 1, bias_attr=False),
            nn.Sigmoid())

    def forward_single_frame(self, x):
        return self.aspp1(x) * self.aspp2(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


if __name__ == '__main__':
    import ppmatting
    backbone = ppmatting.models.MobileNetV3_large_x1_0_os16(
        pretrained='mobilenetv3_large_x1_0_ssld/model.pdparams',
        out_index=[0, 2, 4],
        return_last_conv=True)
    model = RVM(backbone=backbone)
    print(model)

    img = paddle.rand(shape=(1, 3, 512, 512))
    model(img)
