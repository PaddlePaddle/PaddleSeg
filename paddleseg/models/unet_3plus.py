# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddleseg.cvlibs import manager
from paddleseg.models.layers.layer_libs import SyncBatchNorm
from paddleseg.cvlibs.param_init import kaiming_normal_init


@manager.MODELS.add_component
class UNet3Plus(nn.Layer):
    """
    The UNet3+ implementation based on PaddlePaddle.

    The original article refers to
    Huang H , Lin L , Tong R , et al. "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation"
    (https://arxiv.org/abs/2004.08790).

    Args:
        in_channels (int, optional): The channel number of input image.  Default: 3.
        num_classes (int, optional): The unique number of target classes.  Default: 2.
        is_batchnorm (bool, optional): Use batchnorm after conv or not.  Default: True.
        is_deepsup (bool, optional): Use deep supervision or not.  Default: False.
        is_CGM (bool, optional): Use classification-guided module or not.
            If True, is_deepsup must be True.  Default: False.
    """
    def __init__(self, in_channels=3, num_classes=2, is_batchnorm=True, is_deepsup=False, is_CGM=False):
        super(UNet3Plus, self).__init__()
        # parameters
        self.is_deepsup = True if is_CGM else is_deepsup
        self.is_CGM = is_CGM
        # internal definition
        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = 5
        self.up_channels = self.cat_channels * self.cat_blocks
        # layers
        self.encoder = Encoder(in_channels, self.filters, is_batchnorm)
        self.decoder = Decoder(self.filters, self.cat_channels, self.up_channels)
        if self.is_deepsup:
            self.deepsup = DeepSup(self.up_channels, self.filters, num_classes)
            if self.is_CGM:
                self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Conv2D(self.filters[4], 2, 1),
                                         nn.AdaptiveMaxPool2D(1),
                                         nn.Sigmoid())
        else:
            self.outconv1 = nn.Conv2D(self.up_channels, num_classes, 3, padding=1)
        # initialise weights
        for sublayer in self.sublayers ():
            if isinstance(sublayer, nn.Conv2D):
                kaiming_normal_init(sublayer.weight)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                kaiming_normal_init(sublayer.weight)
    def dotProduct(self, seg, cls):
        B, N, H, W = seg.shape
        seg = seg.reshape((B, N, H * W))
        # final = torch.einsum("ijk,ij->ijk", [seg, cls])  # replace it with the following three lines
        clssp = paddle.ones([1, N])
        ecls = (cls * clssp).reshape([B, N, 1])
        final = seg * ecls
        final = final.reshape((B, N, H, W))
        return final
    def forward(self, inputs):
        hs = self.encoder(inputs)
        hds = self.decoder(hs)
        if self.is_deepsup:
            out = self.deepsup(hds)
            if self.is_CGM:
                # classification-guided module
                cls_branch = self.cls(hds[-1]).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
                cls_branch_max = cls_branch.argmax(axis=1)
                cls_branch_max = cls_branch_max.reshape((-1, 1)).astype('float')
                out = [self.dotProduct(d, cls_branch_max) for d in out]
        else:
            out = [self.outconv1(hds[0])]  # d1->320*320*num_classes
        return out


class Encoder(nn.Layer):
    def __init__(self, in_channels, filters, is_batchnorm):
        super(Encoder, self).__init__()
        self.conv1 = UnetConv2D(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2D(kernel_size=2)
        self.conv2 = UnetConv2D(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2)
        self.conv3 = UnetConv2D(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2D(kernel_size=2)
        self.conv4 = UnetConv2D(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2D(kernel_size=2)
        self.conv5 = UnetConv2D(filters[3], filters[4], is_batchnorm)
    def forward(self, inputs):
        h1 = self.conv1(inputs)  # h1->320*320*64
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128
        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256
        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512
        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024
        return [h1, h2, h3, h4, hd5]


class Decoder(nn.Layer):
    def __init__(self, filters, cat_channels, up_channels):
        super(Decoder, self).__init__()
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2D(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2D(filters[0], cat_channels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm(cat_channels)
        self.h1_PT_hd4_relu = nn.ReLU()
        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2D(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2D(filters[1], cat_channels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm(cat_channels)
        self.h2_PT_hd4_relu = nn.ReLU()
        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2D(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2D(filters[2], cat_channels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm(cat_channels)
        self.h3_PT_hd4_relu = nn.ReLU()
        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2D(filters[3], cat_channels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm(cat_channels)
        self.h4_Cat_hd4_relu = nn.ReLU()
        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2D(filters[4], cat_channels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm(cat_channels)
        self.hd5_UT_hd4_relu = nn.ReLU()
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2D(up_channels, up_channels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm(up_channels)
        self.relu4d_1 = nn.ReLU()
        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2D(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2D(filters[0], cat_channels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm(cat_channels)
        self.h1_PT_hd3_relu = nn.ReLU()
        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2D(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2D(filters[1], cat_channels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm(cat_channels)
        self.h2_PT_hd3_relu = nn.ReLU()
        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2D(filters[2], cat_channels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm(cat_channels)
        self.h3_Cat_hd3_relu = nn.ReLU()
        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm(cat_channels)
        self.hd4_UT_hd3_relu = nn.ReLU()
        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2D(filters[4], cat_channels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm(cat_channels)
        self.hd5_UT_hd3_relu = nn.ReLU()
        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2D(up_channels, up_channels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm(up_channels)
        self.relu3d_1 = nn.ReLU()
        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2D(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2D(filters[0], cat_channels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm(cat_channels)
        self.h1_PT_hd2_relu = nn.ReLU()
        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2D(filters[1], cat_channels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm(cat_channels)
        self.h2_Cat_hd2_relu = nn.ReLU()
        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm(cat_channels)
        self.hd3_UT_hd2_relu = nn.ReLU()
        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm(cat_channels)
        self.hd4_UT_hd2_relu = nn.ReLU()
        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2D(filters[4], cat_channels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm(cat_channels)
        self.hd5_UT_hd2_relu = nn.ReLU()
        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.Conv2D_1 = nn.Conv2D(up_channels, up_channels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm(up_channels)
        self.relu2d_1 = nn.ReLU()
        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2D(filters[0], cat_channels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm(cat_channels)
        self.h1_Cat_hd1_relu = nn.ReLU()
        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm(cat_channels)
        self.hd2_UT_hd1_relu = nn.ReLU()
        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm(cat_channels)
        self.hd3_UT_hd1_relu = nn.ReLU()
        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2D(up_channels, cat_channels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm(cat_channels)
        self.hd4_UT_hd1_relu = nn.ReLU()
        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2D(filters[4], cat_channels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm(cat_channels)
        self.hd5_UT_hd1_relu = nn.ReLU()
        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2D(up_channels, up_channels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm(up_channels)
        self.relu1d_1 = nn.ReLU()
    def forward(self, inputs):
        h1, h2, h3, h4, hd5 = inputs
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            paddle.concat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], 1)))) # hd4->40*40*up_channels
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            paddle.concat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], 1)))) # hd3->80*80*up_channels
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.Conv2D_1(
            paddle.concat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], 1)))) # hd2->160*160*up_channels
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            paddle.concat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], 1)))) # hd1->320*320*up_channels
        return [hd1, hd2, hd3, hd4, hd5]


class DeepSup(nn.Layer):
    def __init__(self, up_channels, filters, num_classes):
        super(DeepSup, self).__init__()
        # bilinear upsampling
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # deepsup
        self.outconv1 = nn.Conv2D(up_channels, num_classes, 3, padding=1)
        self.outconv2 = nn.Conv2D(up_channels, num_classes, 3, padding=1)
        self.outconv3 = nn.Conv2D(up_channels, num_classes, 3, padding=1)
        self.outconv4 = nn.Conv2D(up_channels, num_classes, 3, padding=1)
        self.outconv5 = nn.Conv2D(filters[4], num_classes, 3, padding=1)
    def forward(self, inputs):
        hd1, hd2, hd3, hd4, hd5 = inputs
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256
        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256
        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256
        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256
        d1 = self.outconv1(hd1) # 256
        return [d1, d2, d3, d4, d5]


class UnetConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, is_batchnorm, num_conv=2, kernel_size=3, stride=1, padding=1):
        super(UnetConv2D, self).__init__()
        self.num_conv = num_conv
        for i in range(num_conv):
            conv = (nn.Sequential(nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm(out_channels),
                                  nn.ReLU()) \
                    if is_batchnorm else \
                    nn.Sequential(nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.ReLU()))        
            setattr(self, 'conv%d' % (i + 1), conv)
            in_channels = out_channels
        # initialise the blocks
        for children in self.children():
            children.weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
            children.bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
    def forward(self, inputs):
        x = inputs
        for i in range(self.num_conv):
            conv = getattr(self, 'conv%d' % (i + 1))
            x = conv(x)
        return x