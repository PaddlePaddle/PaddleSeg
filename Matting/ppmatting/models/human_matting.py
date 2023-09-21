# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from collections import defaultdict
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg
from paddleseg.models import layers
from paddleseg import utils
from paddleseg.cvlibs import manager

from ppmatting.models.losses import MRSD


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        layers.ConvBNReLU(
            in_channels, out_channels, 3, padding=1),
        nn.Upsample(
            scale_factor=up_sample, mode='bilinear', align_corners=False))


@manager.MODELS.add_component
class HumanMatting(nn.Layer):
    """A model for """

    def __init__(self,
                 backbone,
                 pretrained=None,
                 backbone_scale=0.25,
                 refine_kernel_size=3,
                 if_refine=True):
        super().__init__()
        if if_refine:
            if backbone_scale > 0.5:
                raise ValueError(
                    'Backbone_scale should not be greater than 1/2, but it is {}'
                    .format(backbone_scale))
        else:
            backbone_scale = 1

        self.backbone = backbone
        self.backbone_scale = backbone_scale
        self.pretrained = pretrained
        self.if_refine = if_refine
        if if_refine:
            self.refiner = Refiner(kernel_size=refine_kernel_size)
        self.loss_func_dict = None

        self.backbone_channels = backbone.feat_channels
        ######################
        ### Decoder part - Glance
        ######################
        self.psp_module = layers.PPModule(
            self.backbone_channels[-1],
            512,
            bin_sizes=(1, 3, 5),
            dim_reduction=False,
            align_corners=False)
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)
        # stage 5g
        self.decoder5_g = nn.Sequential(
            layers.ConvBNReLU(
                512 + self.backbone_channels[-1], 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(
                512, 256, 3, padding=2, dilation=2),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4g
        self.decoder4_g = nn.Sequential(
            layers.ConvBNReLU(
                512, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 128, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3g
        self.decoder3_g = nn.Sequential(
            layers.ConvBNReLU(
                256, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2g
        self.decoder2_g = nn.Sequential(
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1g
        self.decoder1_g = nn.Sequential(
            layers.ConvBNReLU(
                128, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0g
        self.decoder0_g = nn.Sequential(
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            nn.Conv2D(
                64, 3, 3, padding=1))

        ##########################
        ### Decoder part - FOCUS
        ##########################
        self.bridge_block = nn.Sequential(
            layers.ConvBNReLU(
                self.backbone_channels[-1], 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(
                512, 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(
                512, 512, 3, dilation=2, padding=2))
        # stage 5f
        self.decoder5_f = nn.Sequential(
            layers.ConvBNReLU(
                512 + self.backbone_channels[-1], 512, 3, padding=1),
            layers.ConvBNReLU(
                512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(
                512, 256, 3, padding=2, dilation=2),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4f
        self.decoder4_f = nn.Sequential(
            layers.ConvBNReLU(
                256 + self.backbone_channels[-2], 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 256, 3, padding=1),
            layers.ConvBNReLU(
                256, 128, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3f
        self.decoder3_f = nn.Sequential(
            layers.ConvBNReLU(
                128 + self.backbone_channels[-3], 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2f
        self.decoder2_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-4], 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 128, 3, padding=1),
            layers.ConvBNReLU(
                128, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1f
        self.decoder1_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-5], 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0f
        self.decoder0_f = nn.Sequential(
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            layers.ConvBNReLU(
                64, 64, 3, padding=1),
            nn.Conv2D(
                64, 1 + 1 + 32, 3, padding=1))
        self.init_weight()

    def forward(self, data):
        src = data['img']
        src_h, src_w = paddle.shape(src)[2:]
        if self.if_refine:
            # It is not need when exporting.
            if isinstance(src_h, paddle.Tensor):
                if (src_h % 4 != 0) or (src_w % 4) != 0:
                    raise ValueError(
                        'The input image must have width and height that are divisible by 4'
                    )

        # Downsample src for backbone
        src_sm = F.interpolate(
            src,
            scale_factor=self.backbone_scale,
            mode='bilinear',
            align_corners=False)

        # Base
        fea_list = self.backbone(src_sm)
        ##########################
        ### Decoder part - GLANCE
        ##########################
        #psp: N, 512, H/32, W/32
        psp = self.psp_module(fea_list[-1])
        #d6_g: N, 512, H/16, W/16
        d5_g = self.decoder5_g(paddle.concat((psp, fea_list[-1]), 1))
        #d5_g: N, 512, H/8, W/8
        d4_g = self.decoder4_g(paddle.concat((self.psp4(psp), d5_g), 1))
        #d4_g: N, 256, H/4, W/4
        d3_g = self.decoder3_g(paddle.concat((self.psp3(psp), d4_g), 1))
        #d4_g: N, 128, H/2, W/2
        d2_g = self.decoder2_g(paddle.concat((self.psp2(psp), d3_g), 1))
        #d2_g: N, 64, H, W
        d1_g = self.decoder1_g(paddle.concat((self.psp1(psp), d2_g), 1))
        #d0_g: N, 3, H, W
        d0_g = self.decoder0_g(d1_g)
        # The 1st channel is foreground. The 2nd is transition region. The 3rd is background.
        # glance_sigmoid = F.sigmoid(d0_g)
        glance_sigmoid = F.softmax(d0_g, axis=1)

        ##########################
        ### Decoder part - FOCUS
        ##########################
        bb = self.bridge_block(fea_list[-1])
        #bg: N, 512, H/32, W/32
        d5_f = self.decoder5_f(paddle.concat((bb, fea_list[-1]), 1))
        #d5_f: N, 256, H/16, W/16
        d4_f = self.decoder4_f(paddle.concat((d5_f, fea_list[-2]), 1))
        #d4_f: N, 128, H/8, W/8
        d3_f = self.decoder3_f(paddle.concat((d4_f, fea_list[-3]), 1))
        #d3_f: N, 64, H/4, W/4
        d2_f = self.decoder2_f(paddle.concat((d3_f, fea_list[-4]), 1))
        #d2_f: N, 64, H/2, W/2
        d1_f = self.decoder1_f(paddle.concat((d2_f, fea_list[-5]), 1))
        #d1_f: N, 64, H, W
        d0_f = self.decoder0_f(d1_f)
        #d0_f: N, 1, H, W
        focus_sigmoid = F.sigmoid(d0_f[:, 0:1, :, :])
        pha_sm = self.fusion(glance_sigmoid, focus_sigmoid)
        err_sm = d0_f[:, 1:2, :, :]
        err_sm = paddle.clip(err_sm, 0., 1.)
        hid_sm = F.relu(d0_f[:, 2:, :, :])

        # Refiner
        if self.if_refine:
            pha = self.refiner(
                src=src, pha=pha_sm, err=err_sm, hid=hid_sm, tri=glance_sigmoid)
            # Clamp outputs
            pha = paddle.clip(pha, 0., 1.)

        if self.training:
            logit_dict = {
                'glance': glance_sigmoid,
                'focus': focus_sigmoid,
                'fusion': pha_sm,
                'error': err_sm
            }
            if self.if_refine:
                logit_dict['refine'] = pha
            loss_dict = self.loss(logit_dict, data)
            return logit_dict, loss_dict
        else:
            return pha if self.if_refine else pha_sm

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            if self.loss_func_dict is None:
                self.loss_func_dict = defaultdict(list)
                self.loss_func_dict['glance'].append(nn.NLLLoss())
                self.loss_func_dict['focus'].append(MRSD())
                self.loss_func_dict['cm'].append(MRSD())
                self.loss_func_dict['err'].append(paddleseg.models.MSELoss())
                self.loss_func_dict['refine'].append(paddleseg.models.L1Loss())
        else:
            self.loss_func_dict = loss_func_dict

        loss = {}

        # glance loss computation
        # get glance label
        glance_label = F.interpolate(
            label_dict['trimap'],
            logit_dict['glance'].shape[2:],
            mode='nearest',
            align_corners=False)
        glance_label_trans = (glance_label == 128).astype('int64')
        glance_label_bg = (glance_label == 0).astype('int64')
        glance_label = glance_label_trans + glance_label_bg * 2
        loss_glance = self.loss_func_dict['glance'][0](
            paddle.log(logit_dict['glance'] + 1e-6), glance_label.squeeze(1))
        loss['glance'] = loss_glance

        # focus loss computation
        focus_label = F.interpolate(
            label_dict['alpha'],
            logit_dict['focus'].shape[2:],
            mode='bilinear',
            align_corners=False)
        loss_focus = self.loss_func_dict['focus'][0](
            logit_dict['focus'], focus_label, glance_label_trans)
        loss['focus'] = loss_focus

        # collaborative matting loss
        loss_cm_func = self.loss_func_dict['cm']
        # fusion_sigmoid loss
        loss_cm = loss_cm_func[0](logit_dict['fusion'], focus_label)
        loss['cm'] = loss_cm

        # error loss
        err = F.interpolate(
            logit_dict['error'],
            label_dict['alpha'].shape[2:],
            mode='bilinear',
            align_corners=False)
        err_label = (F.interpolate(
            logit_dict['fusion'],
            label_dict['alpha'].shape[2:],
            mode='bilinear',
            align_corners=False) - label_dict['alpha']).abs()
        loss_err = self.loss_func_dict['err'][0](err, err_label)
        loss['err'] = loss_err

        loss_all = 0.25 * loss_glance + 0.25 * loss_focus + 0.25 * loss_cm + loss_err

        # refine loss
        if self.if_refine:
            loss_refine = self.loss_func_dict['refine'][0](logit_dict['refine'],
                                                           label_dict['alpha'])
            loss['refine'] = loss_refine
            loss_all = loss_all + loss_refine

        loss['all'] = loss_all
        return loss

    def fusion(self, glance_sigmoid, focus_sigmoid):
        # glance_sigmoid [N, 3, H, W].
        # In index, 0 is foreground, 1 is transition, 2 is backbone.
        # After fusion, the foreground is 1, the background is 0, and the transion is between (0, 1).
        index = paddle.argmax(glance_sigmoid, axis=1, keepdim=True)
        transition_mask = (index == 1).astype('float32')
        fg = (index == 0).astype('float32')
        fusion_sigmoid = focus_sigmoid * transition_mask + fg
        return fusion_sigmoid

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Refiner(nn.Layer):
    '''
    Refiner refines the coarse output to full resolution.

    Args:
        kernel_size: The convolution kernel_size. Options: [1, 3]. Default: 3.
    '''

    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size not in [1, 3]:
            raise ValueError("kernel_size must be in [1, 3]")

        self.kernel_size = kernel_size

        channels = [32, 24, 16, 12, 1]
        self.conv1 = layers.ConvBNReLU(
            channels[0] + 4 + 3,
            channels[1],
            kernel_size,
            padding=0,
            bias_attr=False)
        self.conv2 = layers.ConvBNReLU(
            channels[1], channels[2], kernel_size, padding=0, bias_attr=False)
        self.conv3 = layers.ConvBNReLU(
            channels[2] + 3,
            channels[3],
            kernel_size,
            padding=0,
            bias_attr=False)
        self.conv4 = nn.Conv2D(
            channels[3], channels[4], kernel_size, padding=0, bias_attr=True)

    def forward(self, src, pha, err, hid, tri):
        '''
        Args：
            src: (B, 3, H, W) full resolution source image.
            pha: (B, 1, Hc, Wc) coarse alpha prediction.
            err: (B, 1, Hc, Hc) coarse error prediction.
            hid: (B, 32, Hc, Hc) coarse hidden encoding.
            tri: (B, 1, Hc, Hc) trimap prediction.
        '''
        h_full, w_full = paddle.shape(src)[2:]
        h_half, w_half = h_full // 2, w_full // 2
        h_quat, w_quat = h_full // 4, w_full // 4

        x = paddle.concat([hid, pha, tri], axis=1)
        x = F.interpolate(
            x,
            paddle.stack((h_half, w_half)).squeeze(),
            mode='bilinear',
            align_corners=False)
        y = F.interpolate(
            src,
            paddle.stack((h_half, w_half)).squeeze(),
            mode='bilinear',
            align_corners=False)

        if self.kernel_size == 3:
            x = F.pad(x, [3, 3, 3, 3])
            y = F.pad(y, [3, 3, 3, 3])

        x = self.conv1(paddle.concat([x, y], axis=1))
        x = self.conv2(x)

        if self.kernel_size == 3:
            x = F.interpolate(x, paddle.stack((h_full + 4, w_full + 4)).squeeze())
            y = F.pad(src, [2, 2, 2, 2])
        else:
            x = F.interpolate(
                x, paddle.stack((h_full, w_full)).squeeze(), mode='nearest')
            y = src

        x = self.conv3(paddle.concat([x, y], axis=1))
        x = self.conv4(x)

        pha = x
        return pha
