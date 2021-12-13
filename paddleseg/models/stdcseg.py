# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class STDCSeg(nn.Layer):
    """
    The STDCSeg implementation based on PaddlePaddle.

    The original article refers to Meituan
    Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation."
    (https://arxiv.org/abs/2104.13188)

    Args:
        num_classes(int,optional): The unique number of target classes.
        backbone(nn.Layer): Backbone network, STDCNet1446/STDCNet813. STDCNet1446->STDC2,STDCNet813->STDC813.
        use_boundary_8(bool,non-optional): Whether to use detail loss. it should be True accroding to paper for best metric. Default: True.
        Actually,if you want to use _boundary_2/_boundary_4/_boundary_16,you should append loss function number of DetailAggregateLoss.It should work properly.
        use_conv_last(bool,optional): Determine ContextPath 's inplanes variable according to whether to use bockbone's last conv. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=True,
                 use_boundary_16=False,
                 use_conv_last=False,
                 pretrained=None):
        super(STDCSeg, self).__init__()

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)
        self.ffm = FeatureFusionModule(384, 256)
        self.conv_out = SegHead(256, 256, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out16 = self.conv_out16(feat_cp16)

            logit_list = [feat_out, feat_out8, feat_out16]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegHead, self).__init__()
        self.conv = layers.ConvBNReLU(
            in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class ARM(nn.Layer):
    def __init__(self, in_chan, out_chan):
        super(ARM, self).__init__()
        self.conv = layers.ConvBNReLU(
            in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = paddle.multiply(feat, atten)
        return out


class ContextPath(nn.Layer):
    def __init__(self, backbone, use_conv_last=False):
        super(ContextPath, self).__init__()
        self.backbone = backbone
        self.arm16 = ARM(512, 128)
        inplanes = 1024
        if use_conv_last:
            inplanes = 1024
        self.arm32 = ARM(inplanes, 128)
        self.conv_head32 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = layers.ConvBNReLU(
            inplanes, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        feat8_hw = paddle.shape(feat8)[2:]
        feat16_hw = paddle.shape(feat16)[2:]
        feat32_hw = paddle.shape(feat32)[2:]

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32_hw, mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, feat16_hw, mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8_hw, mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16


class FeatureFusionModule(nn.Layer):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = layers.ConvBNReLU(
            in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2D(
            out_chan,
            out_chan // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None)
        self.conv2 = nn.Conv2D(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = paddle.concat([fsp, fcp], axis=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ARM_0(nn.Layer):
    '''no attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)

    def forward(self, x, y):
        x = self.conv(x)
        out = x + y
        return out


class ARM_0_1(nn.Layer):
    '''no attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        k1 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        out = self.k1 * x + self.k2 * y
        return out


class ARM_0_2(nn.Layer):
    '''no attention'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        k1 = self.create_parameter([out_chan, 1, 1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([out_chan, 1, 1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        out = self.k1 * x + self.k2 * y
        return out


class ARM_1(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = paddle.multiply(x, atten)
        out = x + y
        return out


class ARM_1_1(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = paddle.multiply(x, atten)
        out = self.k1 * x + self.k2 * y
        return out


class ARM_1_2(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = x * self.k1 + (x * atten) * self.k2
        out = x + y
        return out


class ARM_1_3(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([1])
        k2 = self.create_parameter([1])
        k3 = self.create_parameter([1])
        k4 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        self.add_parameter("k2", k2)
        self.add_parameter("k3", k3)
        self.add_parameter("k4", k4)

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = x * self.k1 + (x * atten) * self.k2
        out = self.k3 * x + self.k4 * y
        return out


class ARM_1_4(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = x + paddle.multiply(x, atten)
        out = x + y
        return out


class ARM_1_5(nn.Layer):
    '''use x to attention x (base)'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([out_chan, 1, 1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([out_chan, 1, 1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = paddle.multiply(x, atten)
        out = self.k1 * x + self.k2 * y
        return out


class ARM_2(nn.Layer):
    '''use y to attention y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            y_chan, y_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(y_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(y, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        y = paddle.multiply(y, atten)
        out = x + y
        return out


class ARM_2_1(nn.Layer):
    '''use y to attention y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)

        self.conv_atten = nn.Conv2D(
            y_chan, y_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(y_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)

        atten = F.adaptive_avg_pool2d(y, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        y = paddle.multiply(y, atten)

        out = self.k1 * x + self.k2 * y
        return out


class ARM_3(nn.Layer):
    '''use x to attention x and y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = paddle.multiply(x, atten) + paddle.multiply(y, 1 - atten)
        return out


class ARM_4(nn.Layer):
    '''use y to attention x and y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            y_chan, y_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(y_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(y, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = paddle.multiply(x, atten) + paddle.multiply(y, 1 - atten)
        return out


class ARM_5(nn.Layer):
    '''use x + y to attention x + y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        xy = x + y
        atten = F.adaptive_avg_pool2d(xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = paddle.multiply(xy, atten)
        return out


class ARM_6(nn.Layer):
    '''use x + y to attention x and y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        xy = x + y
        atten = F.adaptive_avg_pool2d(xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = paddle.multiply(x, atten) + paddle.multiply(y, 1 - atten)
        return out


class ARM_7(nn.Layer):
    '''use cat(x,y) to attention x and y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = x * atten + y * (1 - atten)
        return out


class ARM_7_1(nn.Layer):
    '''the afm of attanet'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten_1 = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.conv_atten_2 = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)
        cat_xy = paddle.concat([x, y], axis=1)
        atten = self.conv_atten_1(cat_xy)
        atten = F.adaptive_avg_pool2d(atten, 1)
        atten = self.conv_atten_2(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = x * atten + y * (1 - atten)
        return out


class ARM_7_2(nn.Layer):
    '''add atten and shift'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        k1 = self.create_parameter([1])
        self.add_parameter("k1", k1)
        k2 = self.create_parameter([1])
        self.add_parameter("k2", k2)

    def forward(self, x, y):
        x = self.conv(x)
        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = (x * atten) * self.k1 + (y * (1 - atten)) * self.k2
        return out


class ARM_8(nn.Layer):
    '''adjust conv for x'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = self.conv(x)
        out = paddle.multiply(x, atten) + paddle.multiply(y, 1 - atten)
        return out


class ARM_9(nn.Layer):
    '''combine atten in channel and spatic'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)

        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)

        x_mean = paddle.mean(x, axis=1, keepdim=True)
        y_mean = paddle.mean(y, axis=1, keepdim=True)
        out = x_mean * atten * x + y_mean * (1 - atten) * y
        return out


class ARM_9_1(nn.Layer):
    '''combine atten in channel and spatic'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)

        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        self.conv_x = nn.Conv2D(
            out_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_x = nn.BatchNorm2D(1)
        self.conv_y = nn.Conv2D(
            y_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_y = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)

        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)

        xf = self.bn_x(self.conv_x(x))
        yf = self.bn_y(self.conv_y(y))
        out = xf * atten * x + yf * (1 - atten) * y
        return out


class ARM_10(nn.Layer):
    '''use cat(x,y) to attention x and y'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, out_chan, kernel_size=1, bias_attr=None)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv(x)

        cat_xy = paddle.concat([x, y], axis=1)
        atten = F.adaptive_avg_pool2d(cat_xy, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)

        x_max = paddle.max(x, axis=1, keepdim=True)
        y_max = paddle.max(y, axis=1, keepdim=True)
        out = x_max * atten * x + y_max * (1 - atten) * y
        return out


class ARM_11(nn.Layer):
    '''add spatic atten by mean'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)

    def forward(self, x, y):
        x = self.conv(x)
        xf = paddle.mean(x, axis=1, keepdim=True)
        yf = paddle.mean(y, axis=1, keepdim=True)
        out = xf * x + yf * y
        return out


class ARM_11_1(nn.Layer):
    '''add spatic atten by max'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)

    def forward(self, x, y):
        x = self.conv(x)
        xf = paddle.max(x, axis=1, keepdim=True)
        yf = paddle.max(y, axis=1, keepdim=True)
        out = xf * x + yf * y
        return out


class ARM_11_2(nn.Layer):
    '''add spatic atten by conv'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_x = nn.Conv2D(
            out_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_x = nn.BatchNorm2D(1)
        self.conv_y = nn.Conv2D(
            y_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_y = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)
        xf = self.bn_x(self.conv_x(x))
        yf = self.bn_y(self.conv_y(y))
        out = xf * x + yf * y
        return out


class ARM_11_3(nn.Layer):
    '''add spatic atten by conv and sigmoid'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_x = nn.Conv2D(
            out_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_x = nn.BatchNorm2D(1)
        self.conv_y = nn.Conv2D(
            y_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_y = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)
        xf = F.sigmoid(self.bn_x(self.conv_x(x)))
        yf = F.sigmoid(self.bn_y(self.conv_y(y)))
        out = xf * x + yf * y
        return out


class ARM_11_4(nn.Layer):
    '''add spatic atten by conv'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_x = nn.Conv2D(2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_x = nn.BatchNorm2D(1)
        self.conv_y = nn.Conv2D(2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_y = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)

        x_mean = paddle.mean(x, axis=1, keepdim=True)
        x_max = paddle.max(x, axis=1, keepdim=True)
        x_cat = paddle.concat([x_mean, x_max], axis=1)
        xf = self.bn_x(self.conv_x(x_cat))

        y_mean = paddle.mean(y, axis=1, keepdim=True)
        y_max = paddle.max(y, axis=1, keepdim=True)
        y_cat = paddle.concat([y_mean, y_max], axis=1)
        yf = self.bn_y(self.conv_y(y_cat))

        out = xf * x + yf * y
        return out


class ARM_11_5(nn.Layer):
    '''add spatic atten by conv'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_x = nn.Conv2D(2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.conv_y = nn.Conv2D(2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv(x)

        x_mean = paddle.mean(x, axis=1, keepdim=True)
        x_max = paddle.max(x, axis=1, keepdim=True)
        x_cat = paddle.concat([x_mean, x_max], axis=1)
        xf = F.sigmoid(self.conv_x(x_cat))

        y_mean = paddle.mean(y, axis=1, keepdim=True)
        y_max = paddle.max(y, axis=1, keepdim=True)
        y_cat = paddle.concat([y_mean, y_max], axis=1)
        yf = F.sigmoid(self.conv_y(y_cat))

        out = xf * x + yf * y
        return out


class ARM_11_6(nn.Layer):
    '''add spatic atten by conv'''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_x = nn.Conv2D(2, 1, kernel_size=7, padding=3, bias_attr=False)
        self.conv_y = nn.Conv2D(2, 1, kernel_size=7, padding=3, bias_attr=False)

    def forward(self, x, y):
        x = self.conv(x)

        x_mean = paddle.mean(x, axis=1, keepdim=True)
        x_max = paddle.max(x, axis=1, keepdim=True)
        x_cat = paddle.concat([x_mean, x_max], axis=1)
        xf = F.sigmoid(self.conv_x(x_cat))

        y_mean = paddle.mean(y, axis=1, keepdim=True)
        y_max = paddle.max(y, axis=1, keepdim=True)
        y_cat = paddle.concat([y_mean, y_max], axis=1)
        yf = F.sigmoid(self.conv_y(y_cat))

        out = xf * x + yf * y
        return out


class ARM_12_1(nn.Layer):
    '''combined spatial attention '''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_atten = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)
        cat_xy = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        atten = self.conv_atten(cat_xy)  # n * 1 * h * w
        atten = self.bn_atten(atten)
        out = x * atten + y * (1 - atten)
        return out


class ARM_12_2(nn.Layer):
    '''combined spatial attention '''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            out_chan + y_chan, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv(x)
        cat_xy = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        atten = self.conv_atten(cat_xy)  # n * 1 * h * w
        atten = F.sigmoid(atten)
        out = x * atten + y * (1 - atten)
        return out


class ARM_12_3(nn.Layer):
    '''combined spatial attention '''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_atten = nn.BatchNorm2D(1)

    def forward(self, x, y):
        x = self.conv(x)

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        mean_max_cat = paddle.concat([xy_mean, xy_max], axis=1)
        atten = self.bn_atten(self.conv_atten(mean_max_cat))  # n * 1 * h * w

        out = x * atten + y * (1 - atten)
        return out


class ARM_12_4(nn.Layer):
    '''combined spatial attention '''

    def __init__(self, x_chan, y_chan, out_chan, first_ksize=3):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            x_chan,
            out_chan,
            kernel_size=first_ksize,
            stride=1,
            padding=first_ksize // 2)
        self.conv_atten = nn.Conv2D(
            2, 1, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x, y):
        x = self.conv(x)

        xy_cat = paddle.concat([x, y], axis=1)  # n * 2c * h * w
        xy_mean = paddle.mean(xy_cat, axis=1, keepdim=True)
        xy_max = paddle.max(xy_cat, axis=1, keepdim=True)
        mean_max_cat = paddle.concat([xy_mean, xy_max], axis=1)
        atten = F.sigmoid(self.conv_atten(mean_max_cat))  # n * 1 * h * w

        out = x * atten + y * (1 - atten)
        return out


class ContextPath_pp_0(nn.Layer):
    """The base cpp from stdcseg"""

    def __init__(self,
                 backbone,
                 arm_type,
                 use_conv_last=False,
                 resize_mode='nearest'):
        super().__init__()

        support_backbone = ["STDCNet_pp_0"]
        assert backbone.__class__.__name__ in support_backbone
        self.backbone = backbone
        self.resize_mode = resize_mode

        arm = eval(arm_type)
        print("arm type: " + arm_type)

        inplanes = 1024
        self.arm32 = arm(inplanes, 128, 128, first_ksize=3)
        self.arm16 = arm(512, 128, 128, first_ksize=3)

        self.conv_head32 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = layers.ConvBNReLU(
            inplanes, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        feat8_hw = paddle.shape(feat8)[2:]
        feat16_hw = paddle.shape(feat16)[2:]
        feat32_hw = paddle.shape(feat32)[2:]

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32_hw, mode=self.resize_mode)

        feat32_sum = self.arm32(feat32, avg_up)
        feat32_up = F.interpolate(feat32_sum, feat16_hw, mode=self.resize_mode)
        feat32_up = self.conv_head32(feat32_up)

        feat16_sum = self.arm16(feat16, feat32_up)
        feat16_up = F.interpolate(feat16_sum, feat8_hw, mode=self.resize_mode)
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16


class ContextPath_pp_0_sf(nn.Layer):
    """The cpp support the fusion of sf_net"""

    def __init__(self,
                 backbone,
                 arm_type,
                 use_conv_last=False,
                 resize_mode='nearest'):
        super().__init__()

        support_backbone = ["STDCNet_pp_0"]
        assert backbone.__class__.__name__ in support_backbone
        self.backbone = backbone
        self.resize_mode = resize_mode

        arm = eval(arm_type)
        print("arm type: " + arm_type)

        inplanes = 1024
        self.arm32 = arm(inplanes, 128, 128, first_ksize=3)
        self.arm16 = arm(512, 128, 128, first_ksize=3)

        self.conv_head32 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = layers.ConvBNReLU(
            inplanes, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        feat8_hw = paddle.shape(feat8)[2:]
        feat16_hw = paddle.shape(feat16)[2:]
        feat32_hw = paddle.shape(feat32)[2:]

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32_hw, mode=self.resize_mode)

        feat32_sum = self.arm32(feat32, avg_up)
        feat32_up = F.interpolate(feat32_sum, feat16_hw, mode=self.resize_mode)
        feat32_up = self.conv_head32(feat32_up)

        feat16_sum = self.arm16(feat16, feat32_up)
        feat16_up = F.interpolate(feat16_sum, feat8_hw, mode=self.resize_mode)
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16


class ContextPath_pp_1(nn.Layer):
    """Consider all feature map from encoder"""

    def __init__(self,
                 backbone,
                 arm_type,
                 use_conv_last=False,
                 resize_mode='nearest'):
        super().__init__()

        support_backbone = ["STDCNet_pp_1"]
        assert backbone.__class__.__name__ in support_backbone
        self.backbone = backbone
        self.resize_mode = resize_mode

        arm = eval(arm_type)
        print("arm type: " + arm_type)

        inplanes = 1024
        self.arm32 = arm(2 * inplanes, 128, 128, first_ksize=1)
        self.arm16 = arm(2 * 512, 128, 128, first_ksize=1)

        self.conv_avg = layers.ConvBNReLU(
            inplanes, 128, kernel_size=1, stride=1, padding=0)
        self.conv_head16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head32 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x2, x4, x8_1, x8_2, x16_1, x16_2, x32_1, x32_2 = self.backbone(x)
        x16_cat = paddle.concat([x16_1, x16_2], axis=1)
        x32_cat = paddle.concat([x32_1, x32_2], axis=1)

        x8_hw = paddle.shape(x8_1)[2:]
        x16_hw = paddle.shape(x16_1)[2:]
        x32_hw = paddle.shape(x32_1)[2:]

        avg = F.adaptive_avg_pool2d(x32_2, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, x32_hw, mode=self.resize_mode)

        x32_sum = self.arm32(x32_cat, avg_up)
        x32_up = F.interpolate(x32_sum, x16_hw, mode=self.resize_mode)
        x32_up = self.conv_head32(x32_up)

        x16_sum = self.arm16(x16_cat, x32_up)
        x16_up = F.interpolate(x16_sum, x8_hw, mode=self.resize_mode)
        x16_up = self.conv_head16(x16_up)

        return x2, x4, x8_2, x16_2, x16_up, x32_up  # x8, x16


class FeatureFusionModule_process_feat4(nn.Layer):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = layers.ConvBNReLU(
            in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2D(
            out_chan,
            out_chan // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None)
        self.conv2 = nn.Conv2D(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, en_4, de_4):
        fcat = paddle.concat([en_4, de_4], axis=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class FeatureFusionModule_concat_3(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv_de_16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_de_8 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_de_fuse_8 = layers.ConvBNReLU(
            256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, de_16, de_8, de_fuse_8):
        x_hw = paddle.shape(de_fuse_8)[2:]

        de_16 = self.conv_de_16(de_16)
        de_16_up = F.interpolate(de_16, x_hw, mode='bilinear')

        de_8 = self.conv_de_8(de_8)

        de_fuse_8 = self.conv_de_fuse_8(de_fuse_8)

        feat_out = paddle.concat([de_16_up, de_8, de_fuse_8], axis=1)
        return feat_out


class FeatureFusionModule_concat_4(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv_de_16 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_de_8 = layers.ConvBNReLU(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_de_fuse_8 = layers.ConvBNReLU(
            256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, de_16, de_8, de_fuse_8, de_fuse_4):
        x_hw = paddle.shape(de_fuse_4)[2:]

        de_16 = self.conv_de_16(de_16)
        de_16_up = F.interpolate(de_16, x_hw, mode='bilinear')

        de_8 = self.conv_de_8(de_8)
        de_8_up = F.interpolate(de_8, x_hw, mode='bilinear')

        de_fuse_8 = self.conv_de_fuse_8(de_fuse_8)
        de_fuse_8_up = F.interpolate(de_fuse_8, x_hw, mode='bilinear')

        feat_out = paddle.concat([de_16_up, de_8_up, de_fuse_8_up, de_fuse_4],
                                 axis=1)

        return feat_out


class FeatureFusionModule_feat4_pool(FeatureFusionModule):
    def __init__(self, in_chan, out_chan):
        super().__init__(in_chan, out_chan)

    def forward(self, en_4, en_8, de_8):
        en_4_down = F.avg_pool2d(en_4, kernel_size=2, stride=2)
        fcat = paddle.concat([en_4_down, en_8, de_8], axis=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class FeatureFusionModule_feat4_conv(FeatureFusionModule):
    def __init__(self, in_chan, out_chan):
        super().__init__(in_chan, out_chan)

        self.en_4_down = layers.ConvBNReLU(
            64, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, en_4, en_8, de_8):
        en_4_down = self.en_4_down(en_4)
        fcat = paddle.concat([en_4_down, en_8, de_8], axis=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


@manager.MODELS.add_component
class STDCSeg_feat4(STDCSeg):
    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=True,
                 use_boundary_16=False,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__(num_classes, backbone, use_boundary_2, use_boundary_4,
                         use_boundary_8, use_boundary_16, use_conv_last,
                         pretrained)
        self.ffm = None

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse = self.ffm(feat_res4, feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out16 = self.conv_out16(feat_cp16)

            logit_list = [feat_out, feat_out8, feat_out16]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse = self.ffm(feat_res4, feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list


@manager.MODELS.add_component
class STDCSeg_feat4_pool(STDCSeg_feat4):
    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=True,
                 use_boundary_16=False,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__(num_classes, backbone, use_boundary_2, use_boundary_4,
                         use_boundary_8, use_boundary_16, use_conv_last,
                         pretrained)

        self.ffm = FeatureFusionModule_feat4_pool(448, 256)


@manager.MODELS.add_component
class STDCSeg_feat4_conv(STDCSeg_feat4):
    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=True,
                 use_boundary_16=False,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__(num_classes, backbone, use_boundary_2, use_boundary_4,
                         use_boundary_8, use_boundary_16, use_conv_last,
                         pretrained)

        self.ffm = FeatureFusionModule_feat4_conv(448, 256)


@manager.MODELS.add_component
class STDCSeg_pp(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=True,
                 use_boundary_16=False,
                 use_conv_last=False,
                 cpp_type='ContextPath_pp_0',
                 arm_type='ARM_1',
                 resize_mode='nearest',
                 pretrained=None):
        super().__init__()

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        print("cpp type:" + cpp_type)
        self.cp = eval(cpp_type)(backbone, arm_type, use_conv_last, resize_mode)
        self.ffm = FeatureFusionModule(384, 256)
        self.conv_out = SegHead(256, 256, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out16 = self.conv_out16(feat_cp16)

            logit_list = [feat_out, feat_out8, feat_out16]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.MODELS.add_component
class STDCSeg_pseudo_prtrained(STDCSeg):
    def __init__(self,
                 num_classes,
                 backbone,
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=False,
                 use_boundary_16=False,
                 use_conv_last=False,
                 pretrained=None):
        super().__init__(num_classes, backbone, use_boundary_2, use_boundary_4,
                         use_boundary_8, use_boundary_16, use_conv_last,
                         pretrained)

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)

            logit_list = [feat_out]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list


@manager.MODELS.add_component
class STDCSeg_concat_3(nn.Layer):
    def __init__(
            self,
            num_classes,
            backbone,
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=True,
            use_boundary_16=False,
            use_conv_last=False,
            out_flag='f_cat',  # support f8_fuse, f_cat
            pretrained=None):
        super().__init__()

        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)
        self.ffm = FeatureFusionModule(384, 256)
        self.cat = FeatureFusionModule_concat_3()

        self.conv_out_cat = SegHead(512, 256, num_classes)
        self.conv_out_fuse8 = SegHead(256, 256, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out16 = SegHead(128, 64, num_classes)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)

        self.out_flag = out_flag
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse8 = self.ffm(feat_res8, feat_cp8)
            feat_cat = self.cat(feat_cp16, feat_cp8, feat_fuse8)

            feat_out_cat = self.conv_out_cat(feat_cat)
            feat_out_fuse8 = self.conv_out_fuse8(feat_fuse8)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out16 = self.conv_out16(feat_cp16)

            logit_list = [feat_out_cat, feat_out_fuse8, feat_out8, feat_out16]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse8 = self.ffm(feat_res8, feat_cp8)
            if self.out_flag == 'f8_fuse':
                feat_out_fuse8 = self.conv_out_fuse8(feat_fuse8)
                feat_out = feat_out_fuse8
            elif self.out_flag == 'f_cat':
                feat_cat = self.cat(feat_cp16, feat_cp8, feat_fuse8)
                feat_out_cat = self.conv_out_cat(feat_cat)
                feat_out = feat_out_cat

            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.MODELS.add_component
class STDCSeg_fuse_f4(nn.Layer):
    def __init__(
            self,
            num_classes,
            backbone,
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=True,
            use_boundary_16=False,
            use_conv_last=False,
            out_flag='f8_fuse',  # support f8_fuse, f4_fuse, f_cat
            pretrained=None):
        super().__init__()

        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)

        self.ffm8 = FeatureFusionModule(384, 256)
        self.ffm4 = FeatureFusionModule_process_feat4(320, 256)

        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out8_fuse = SegHead(256, 256, num_classes)
        self.conv_out4_fuse = SegHead(256, 256, num_classes)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)

        self.out_flag = out_flag
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            feat_cp4 = F.interpolate(
                feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
            feat_fuse4 = self.ffm4(feat_res4, feat_cp4)

            feat_out16 = self.conv_out16(feat_cp16)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
            feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)

            logit_list = [feat_out4_fuse, feat_out8_fuse, feat_out8, feat_out16]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            if self.out_flag == 'f8_fuse':
                feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
                feat_out = feat_out8_fuse
            elif self.out_flag == 'f4_fuse':
                feat_cp4 = F.interpolate(
                    feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
                feat_fuse4 = self.ffm4(feat_res4, feat_cp4)
                feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)
                feat_out = feat_out4_fuse

            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.MODELS.add_component
class STDCSeg_concat_4(nn.Layer):
    def __init__(
            self,
            num_classes,
            backbone,
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=True,
            use_boundary_16=False,
            use_conv_last=False,
            out_flag='f_cat',  # support f8_fuse, f4_fuse, f_cat
            pretrained=None):
        super().__init__()

        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)

        self.ffm8 = FeatureFusionModule(384, 256)
        self.ffm4 = FeatureFusionModule_process_feat4(320, 256)
        self.ffm_cat = FeatureFusionModule_concat_4()

        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out8_fuse = SegHead(256, 256, num_classes)
        self.conv_out4_fuse = SegHead(256, 256, num_classes)
        self.conv_cat = SegHead(768, 256, num_classes)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)

        self.out_flag = out_flag
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            feat_cp4 = F.interpolate(
                feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
            feat_fuse4 = self.ffm4(feat_res4, feat_cp4)

            feat_cat = self.ffm_cat(feat_cp16, feat_cp8, feat_fuse8, feat_fuse4)

            feat_out16 = self.conv_out16(feat_cp16)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
            feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)
            feat_out_cat = self.conv_cat(feat_cat)

            logit_list = [
                feat_out_cat, feat_out4_fuse, feat_out8_fuse, feat_out8,
                feat_out16
            ]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            if self.out_flag == 'f8_fuse':
                feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
                feat_out = feat_out8_fuse
            elif self.out_flag == 'f4_fuse':
                feat_cp4 = F.interpolate(
                    feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
                feat_fuse4 = self.ffm4(feat_res4, feat_cp4)
                feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)
                feat_out = feat_out4_fuse
            elif self.out_flag == 'f_cat':
                feat_cp4 = F.interpolate(
                    feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
                feat_fuse4 = self.ffm4(feat_res4, feat_cp4)

                feat_cat = self.ffm_cat(feat_cp16, feat_cp8, feat_fuse8,
                                        feat_fuse4)
                feat_out_cat = self.conv_cat(feat_cat)
                feat_out = feat_out_cat

            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.MODELS.add_component
class STDCSeg_concat4_distill(nn.Layer):
    def __init__(
            self,
            num_classes,
            backbone,
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=True,
            use_boundary_16=False,
            use_conv_last=False,
            out_flag='f_cat',  # support f8_fuse, f4_fuse, f_cat
            pretrained=None):
        super().__init__()

        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)

        self.ffm8 = FeatureFusionModule(384, 256)
        self.ffm4 = FeatureFusionModule_process_feat4(320, 256)
        self.ffm_cat = FeatureFusionModule_concat_4()

        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out8_fuse = SegHead(256, 256, num_classes)
        self.conv_out4_fuse = SegHead(256, 256, num_classes)
        self.conv_cat = SegHead(768, 256, num_classes)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)

        self.out_flag = out_flag
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        feat_res2, feat_res4, feat_res8, _, feat_cp8, feat_cp16 = self.cp(x)

        logit_list = []
        if self.training:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            feat_cp4 = F.interpolate(
                feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
            feat_fuse4 = self.ffm4(feat_res4, feat_cp4)

            feat_cat = self.ffm_cat(feat_cp16, feat_cp8, feat_fuse8, feat_fuse4)

            feat_out16 = self.conv_out16(feat_cp16)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
            feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)
            feat_out_cat = self.conv_cat(feat_cat)

            logit_list = [
                feat_out_cat, feat_out4_fuse, feat_out8_fuse, feat_out8,
                feat_out16, feat_out8_fuse
            ]
            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=True)
                for x in logit_list
            ]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse8 = self.ffm8(feat_res8, feat_cp8)

            if self.out_flag == 'f8_fuse':
                feat_out8_fuse = self.conv_out8_fuse(feat_fuse8)
                feat_out = feat_out8_fuse
            elif self.out_flag == 'f4_fuse':
                feat_cp4 = F.interpolate(
                    feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
                feat_fuse4 = self.ffm4(feat_res4, feat_cp4)
                feat_out4_fuse = self.conv_out4_fuse(feat_fuse4)
                feat_out = feat_out4_fuse
            elif self.out_flag == 'f_cat':
                feat_cp4 = F.interpolate(
                    feat_fuse8, paddle.shape(feat_res4)[2:], mode='bilinear')
                feat_fuse4 = self.ffm4(feat_res4, feat_cp4)

                feat_cat = self.ffm_cat(feat_cp16, feat_cp8, feat_fuse8,
                                        feat_fuse4)
                feat_out_cat = self.conv_cat(feat_cat)
                feat_out = feat_out_cat

            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=True)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
