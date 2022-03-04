# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/black0017/MedicalZooPytorch

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

import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import utils


@manager.MODELS.add_component
class UNet3D(nn.Layer):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet3D, self).__init__()
        self.best_loss = 1000000
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3D(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2,
                                   mode='trilinear',
                                   data_format="NCDHW")
        self.softmax = nn.Softmax(axis=1)

        self.conv3d_c1_1 = nn.Conv3D(self.in_channels,
                                     self.base_n_filter,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias_attr=False)
        self.conv3d_c1_2 = nn.Conv3D(self.base_n_filter,
                                     self.base_n_filter,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias_attr=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter,
                                             self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3D(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3D(self.base_n_filter,
                                   self.base_n_filter * 2,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias_attr=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2,
                                                       self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3D(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3D(self.base_n_filter * 2,
                                   self.base_n_filter * 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias_attr=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4,
                                                       self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3D(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3D(self.base_n_filter * 4,
                                   self.base_n_filter * 8,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias_attr=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8,
                                                       self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3D(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3D(self.base_n_filter * 8,
                                   self.base_n_filter * 16,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias_attr=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16,
                                                       self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3D(self.base_n_filter * 8,
                                   self.base_n_filter * 8,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias_attr=False)
        self.inorm3d_l0 = nn.InstanceNorm3D(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16,
                                                       self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3D(self.base_n_filter * 16,
                                   self.base_n_filter * 8,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias_attr=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 8, self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8,
                                                       self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3D(self.base_n_filter * 8,
                                   self.base_n_filter * 4,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias_attr=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 4, self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4,
                                                       self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3D(self.base_n_filter * 4,
                                   self.base_n_filter * 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias_attr=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 2, self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2,
                                                       self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3D(self.base_n_filter * 2,
                                   self.n_classes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias_attr=False)

        self.ds2_1x1_conv3d = nn.Conv3D(self.base_n_filter * 8,
                                        self.n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias_attr=False)
        self.ds3_1x1_conv3d = nn.Conv3D(self.base_n_filter * 4,
                                        self.n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3D(feat_in,
                      feat_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False), nn.InstanceNorm3D(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3D(feat_in), nn.LeakyReLU(),
            nn.Conv3D(feat_in,
                      feat_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3D(feat_in,
                      feat_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3D(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', data_format='NCDHW'),
            # should be feat_in*2 or feat_in
            nn.Conv3D(feat_in,
                      feat_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False),
            nn.InstanceNorm3D(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = paddle.concat([out, context_4], axis=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = paddle.concat([out, context_3], axis=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = paddle.concat([out, context_2], axis=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = paddle.concat([out, context_1], axis=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(
            ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        return out

    def test(self):
        import numpy as np
        np.random.seed(1)
        a = np.random.rand(1, self.in_channels, 32, 32, 32)
        input_tensor = paddle.to_tensor(a, dtype='float32')

        ideal_out = paddle.rand((1, self.n_classes, 32, 32, 32))
        out = self.forward(input_tensor)
        print("out", out.mean(), input_tensor.mean())

        assert ideal_out.shape == out.shape
        paddle.summary(self, (1, self.in_channels, 32, 32, 32))

        print("Vnet test is complete")


if __name__ == "__main__":
    m = UNet3D(in_channels=1, n_classes=2)
    m.test()
