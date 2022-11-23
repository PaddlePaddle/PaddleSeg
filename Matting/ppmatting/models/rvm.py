# TODO add gpl license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
import paddleseg
from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from typing import Tuple, Optional

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

from ppmatting.models import FastGuidedFilter


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
            decoder_channels=(80, 40, 32, 16),
            refiner='deep_guided_filter',
            pretrained=None, ):
        super().__init__()

        self.backbone = backbone
        self.aspp = LRASPP(lraspp_in_channels, lraspp_out_channels)
        rd_fea_channels = self.backbone.feat_channels[:-1] + [
            lraspp_out_channels
        ]
        self.decoder = RecurrentDecoder(rd_fea_channels, decoder_channels)

        self.project_mat = Projection(decoder_channels[-1], 4)
        self.project_seg = Projection(decoder_channels[-1], 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

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
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], axis=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha,
                                                 hid)
            fgr = fgr_residual + src
            fgr = fgr.clip(0., 1.)
            pha = pha.clip(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = paddle.shape(x)[:2]
            x = F.interpolate(
                x.flatten(0, 1),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False)
            *_, C, H, W = paddle.shape(x)[-3:]
            x = x.reshape((B, T, C, H, W))
        else:
            x = F.interpolate(
                x,
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False)
        return x


class LRASPP(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2D(
                in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU())
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


class RecurrentDecoder(nn.Layer):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2],
                                       3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1],
                                       3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0],
                                       3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor,
                f1: Tensor,
                f2: Tensor,
                f3: Tensor,
                f4: Tensor,
                r1: Optional[Tensor],
                r2: Optional[Tensor],
                r3: Optional[Tensor],
                r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4


class AvgPool(nn.Layer):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2D(2, 2, ceil_mode=True)

    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3

    def forward_time_series(self, s0):
        B, T = paddle.shape(s0)[:2]
        s0 = s0.flatten_(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.reshape((B, T, *(paddle.shape(s1)[1:])))
        s2 = s2.reshape((B, T, *(paddle.shape(s2)[1:])))
        s3 = s3.reshape((B, T, *(paddle.shape(s3)[1:])))
        return s1, s2, s3

    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)

    def forward(self, x, r=None):
        a, b = x.split(2, axis=-3)
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=-3)
        return x, r


class UpsamplingBlock(nn.Layer):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels + skip_channels + src_channels,
                out_channels,
                3,
                1,
                1,
                bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(), )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :paddle.shape(s)[2], :paddle.shape(s)[3]]
        x = paddle.concat([x, f, s], axis=1)
        x = self.conv(x)
        a, b = x.split(2, axis=1)
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=1)
        return x, r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = paddle.concat([x, f, s], axis=1)
        x = self.conv(x)
        _, c, h, w = paddle.shape(x)
        x = x.reshape((B, T, c, h, w))
        a, b = x.split(2, axis=2)
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=2)
        return x, r

    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Layer):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels + src_channels,
                out_channels,
                3,
                1,
                1,
                bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(
                out_channels, out_channels, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(), )

    def forward_single_frame(self, x, s):
        _, _, H, W = paddle.shape(s)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = paddle.concat([x, s], axis=1)
        x = self.conv(x)
        return x

    def forward_time_series(self, x, s):
        B, T, C, H, W = paddle.shape(s)
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = paddle.concat([x, s], axis=1)
        x = self.conv(x)
        x = paddle.reshape(x, (B, T, paddle.shape(x)[1], H, W))
        return x

    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Layer):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2D(
                channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid())
        self.hh = nn.Sequential(
            nn.Conv2D(
                channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh())

    def forward_single_frame(self, x, h):
        r, z = self.ih(paddle.concat([x, h], axis=1)).split(2, axis=1)
        c = self.hh(paddle.concat([x, r * h], axis=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(axis=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = paddle.stack(o, axis=1)
        return o, h

    def forward(self, x, h=None):
        if h is None:
            h = paddle.zeros(
                (paddle.shape(x)[0], paddle.shape(x)[-3], paddle.shape(x)[-2],
                 paddle.shape(x)[-1]),
                dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return self.conv(x)

    def forward_time_series(self, x):
        B, T = paddle.shape(x)[:2]
        x = self.conv(x.flatten(0, 1))
        _, C, H, W = paddle.shape(x)
        x = x.reshape((B, T, C, H, W))
        return x

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class FastGuidedFilterRefiner(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)

    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha):
        fine_src_gray = fine_src.mean(1, keepdim=True)
        base_src_gray = base_src.mean(1, keepdim=True)

        fgr, pha = self.guilded_filter(
            paddle.concat(
                [base_src, base_src_gray], axis=1),
            paddle.concat(
                [base_fgr, base_pha], axis=1),
            paddle.concat(
                [fine_src, fine_src_gray], axis=1)).split(
                    [3, 1], axis=1)

        return fgr, pha

    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1), base_pha.flatten(0, 1))
        *_, C, H, W = paddle.shape(fgr)
        fgr = fgr.reshape((B, T, C, H, W))
        pha = pha.reshape((B, T, 1, H, W))
        return fgr, pha

    def forward(self, fine_src, base_src, base_fgr, base_pha, *args, **kwargs):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr,
                                            base_pha)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr,
                                             base_pha)


class DeepGuidedFilterRefiner(nn.Layer):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2D(
            4, 4, kernel_size=3, padding=1, bias_attr=False, groups=4)
        self.box_filter.weight.set_value(
            paddle.zeros_like(self.box_filter.weight) + 1 / 9)
        self.conv = nn.Sequential(
            nn.Conv2D(
                4 * 2 + hid_channels,
                hid_channels,
                kernel_size=1,
                bias_attr=False),
            nn.BatchNorm2D(hid_channels),
            nn.ReLU(),
            nn.Conv2D(
                hid_channels, hid_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hid_channels),
            nn.ReLU(),
            nn.Conv2D(
                hid_channels, 4, kernel_size=1, bias_attr=True))

    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha,
                             base_hid):
        fine_x = paddle.concat(
            [fine_src, fine_src.mean(
                1, keepdim=True)], axis=1)
        base_x = paddle.concat(
            [base_src, base_src.mean(
                1, keepdim=True)], axis=1)
        base_y = paddle.concat([base_fgr, base_pha], axis=1)

        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x = self.box_filter(base_x * base_x) - mean_x * mean_x

        A = self.conv(paddle.concat([cov_xy, var_x, base_hid], axis=1))
        b = mean_y - A * mean_x

        H, W = paddle.shape(fine_src)[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)

        out = A * fine_x + b
        fgr, pha = out.split([3, 1], axis=1)
        return fgr, pha

    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha,
                            base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1), base_hid.flatten(0, 1))
        *_, C, H, W = paddle.shape(fgr)
        fgr = fgr.reshape((B, T, C, H, W))
        pha = pha.reshape((B, T, 1, H, W))
        return fgr, pha

    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr,
                                            base_pha, base_hid)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr,
                                             base_pha, base_hid)


if __name__ == '__main__':
    import ppmatting
    backbone = ppmatting.models.MobileNetV3_large_x1_0_os16(
        pretrained='mobilenetv3_large_x1_0_ssld/model.pdparams',
        out_index=[0, 2, 4],
        return_last_conv=True)
    model = RVM(backbone=backbone, refiner='deep_guider_filter')
    print(model)

    img = paddle.rand(shape=(1, 3, 512, 512))
    result = model(img, downsample_ratio=0.25)

    print('result')
    for i in result:
        print(i.shape)

    #fine_src, base_src, base_fgr, base_pha

    # fine_src = paddle.rand(shape=(1, 3, 512, 512))
    # base_src = paddle.rand(shape=(1, 3, 128, 128))
    # base_fgr = paddle.rand(shape=(1, 3, 128, 128))
    # base_pha = paddle.rand(shape=(1, 1, 128, 128))
    # base_hid = paddle.rand(shape=(1, 16, 128, 128))

    # model = DeepGuidedFilterRefiner()
    # print(model)
    # results = model(fine_src, base_src, base_fgr, base_pha, base_hid)
    # for result in results:
    #     print(result.shape)
