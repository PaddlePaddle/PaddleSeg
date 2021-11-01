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

from model import MRSD


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        layers.ConvBNReLU(in_channels, out_channels, 3, padding=1),
        nn.Upsample(
            scale_factor=up_sample, mode='bilinear', align_corners=False))


@manager.MODELS.add_component
class ZiYan(nn.Layer):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained

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
            layers.ConvBNReLU(512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(512, 256, 3, padding=2, dilation=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4g
        self.decoder4_g = nn.Sequential(
            layers.ConvBNReLU(512, 256, 3, padding=1),
            layers.ConvBNReLU(256, 256, 3, padding=1),
            layers.ConvBNReLU(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3g
        self.decoder3_g = nn.Sequential(
            layers.ConvBNReLU(256, 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2g
        self.decoder2_g = nn.Sequential(
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1g
        self.decoder1_g = nn.Sequential(
            layers.ConvBNReLU(128, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0g
        self.decoder0_g = nn.Sequential(
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, 3, 3, padding=1))

        ##########################
        ### Decoder part - FOCUS
        ##########################
        self.bridge_block = nn.Sequential(
            layers.ConvBNReLU(
                self.backbone_channels[-1], 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(512, 512, 3, dilation=2, padding=2),
            layers.ConvBNReLU(512, 512, 3, dilation=2, padding=2))
        # stage 5f
        self.decoder5_f = nn.Sequential(
            layers.ConvBNReLU(
                512 + self.backbone_channels[-1], 512, 3, padding=1),
            layers.ConvBNReLU(512, 512, 3, padding=2, dilation=2),
            layers.ConvBNReLU(512, 256, 3, padding=2, dilation=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 4f
        self.decoder4_f = nn.Sequential(
            layers.ConvBNReLU(
                256 + self.backbone_channels[-2], 256, 3, padding=1),
            layers.ConvBNReLU(256, 256, 3, padding=1),
            layers.ConvBNReLU(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 3f
        self.decoder3_f = nn.Sequential(
            layers.ConvBNReLU(
                128 + self.backbone_channels[-3], 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 2f
        self.decoder2_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-4], 128, 3, padding=1),
            layers.ConvBNReLU(128, 128, 3, padding=1),
            layers.ConvBNReLU(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 1f
        self.decoder1_f = nn.Sequential(
            layers.ConvBNReLU(
                64 + self.backbone_channels[-5], 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # stage 0f
        self.decoder0_f = nn.Sequential(
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, 1, 3, padding=1))

        self.init_weight()

    def forward(self, inputs):
        x = inputs['img']
        # input fea_list shape [N, 64, H/2, W/2] [N, 64, H/4, W/4]
        # [N, 128, H/8, W/8] [N, 256, H/16, W/16] [N, 512, H/32, W/32]
        fea_list = self.backbone(x)

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
        focus_sigmoid = F.sigmoid(d0_f)

        fusion_sigmoid = self.fusion(glance_sigmoid, focus_sigmoid)

        if self.training:
            logit_dict = {
                'glance': glance_sigmoid,
                'focus': focus_sigmoid,
                'fusion': fusion_sigmoid
            }
            return logit_dict
        else:
            return fusion_sigmoid

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            loss_func_dict = defaultdict(list)
            loss_func_dict['glance'].append(nn.NLLLoss())
            loss_func_dict['focus'].append(MRSD())
            loss_func_dict['cm'].append(MRSD())
            loss_func_dict['cm'].append(MRSD())

        loss = {}

        # glance loss computation
        # get glance label
        glance_label = label_dict['trimap']
        glance_label_trans = (glance_label == 128).astype('int64')
        glance_label_bg = (glance_label == 0).astype('int64')
        glance_label = glance_label_trans + glance_label_bg * 2
        loss_glance = loss_func_dict['glance'][0](
            paddle.log(logit_dict['glance'] + 1e-6), glance_label.squeeze(1))
        loss['glance'] = loss_glance
        # TODO glance label 的验证

        # focus loss computation
        loss_focus = loss_func_dict['focus'][0](logit_dict['focus'],
                                                label_dict['alpha'],
                                                label_dict['trimap'] == 128)
        loss['focus'] = loss_focus

        # collaborative matting loss
        loss_cm_func = loss_func_dict['cm']
        # fusion_sigmoid loss
        loss_cm = loss_cm_func[0](logit_dict['fusion'], label_dict['alpha'])
        # composion loss
        comp_pred = logit_dict['fusion'] * label_dict['fg'] + (
            1 - logit_dict['fusion']) * label_dict['bg']
        loss_cm = loss_cm + loss_cm_func[1](comp_pred, label_dict['img'])
        loss['cm'] = loss_cm

        loss['all'] = 0.25 * loss_glance + 0.25 * loss_focus + 0.25 * loss['cm']

        return loss

    def fusion(self, glance_sigmoid, focus_sigmoid):
        # glance_sigmoid [N, 3, H, W]
        # In index, 0 is foreground, 1 is transition, 2 is backbone
        # After fusion, the foreground is 1, the background is 0, and the transion is between [0, 1]
        index = paddle.argmax(glance_sigmoid, axis=1, keepdim=True)
        transition_mask = (index == 1).astype('float32')
        fg = (index == 0).astype('float32')
        fusion_sigmoid = focus_sigmoid * transition_mask + fg
        return fusion_sigmoid

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.MODELS.add_component
class ZiYanRefine(ZiYan):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 backbone_scale=0.25,
                 refine_mode='sampling',
                 refine_sample_pixels=80000,
                 refine_threshold=0.1,
                 refine_kernel_size=3,
                 refine_prevent_oversampling=True,
                 if_refine=True):
        if if_refine:
            if backbone_scale > 0.5:
                raise ValueError(
                    'Backbone_scale should not be greater than 1/2, but it is {}'
                    .format(backbone_scale))
        else:
            backbone_scale = 1
        super().__init__(backbone)

        self.backbone_scale = backbone_scale
        self.pretrained = pretrained
        self.if_refine = if_refine
        if if_refine:
            self.refiner = Refiner(
                mode=refine_mode,
                sample_pixels=refine_sample_pixels,
                threshold=refine_threshold,
                kernel_size=refine_kernel_size,
                prevent_oversampling=refine_prevent_oversampling)

        # stage 0f recontain
        self.decoder0_f = nn.Sequential(
            layers.ConvBNReLU(64, 64, 3, padding=1),
            layers.ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, 1 + 1 + 32, 3, padding=1))
        self.init_weight()

    def forward(self, data):
        src = data['img']
        src_h, src_w = src.shape[2:]
        if self.if_refine:
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
            pha, ref_sm = self.refiner(
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
            return logit_dict
        else:
            return pha if self.if_refine else pha_sm

    def loss(self, logit_dict, label_dict, loss_func_dict=None):
        if loss_func_dict is None:
            loss_func_dict = defaultdict(list)
            loss_func_dict['glance'].append(nn.NLLLoss())
            loss_func_dict['focus'].append(MRSD())
            loss_func_dict['cm'].append(MRSD())
            loss_func_dict['err'].append(paddleseg.models.MSELoss())
            loss_func_dict['refine'].append(paddleseg.models.L1Loss())

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
        loss_glance = loss_func_dict['glance'][0](
            paddle.log(logit_dict['glance'] + 1e-6), glance_label.squeeze(1))
        loss['glance'] = loss_glance
        # TODO glance label 的验证

        # focus loss computation
        focus_label = F.interpolate(
            label_dict['alpha'],
            logit_dict['focus'].shape[2:],
            mode='bilinear',
            align_corners=False)
        loss_focus = loss_func_dict['focus'][0](logit_dict['focus'],
                                                focus_label, glance_label_trans)
        loss['focus'] = loss_focus

        # collaborative matting loss
        loss_cm_func = loss_func_dict['cm']
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
        loss_err = loss_func_dict['err'][0](err, err_label)
        loss['err'] = loss_err

        loss_all = 0.25 * loss_glance + 0.25 * loss_focus + 0.25 * loss_cm + loss_err

        # refine loss
        if self.if_refine:
            loss_refine = loss_func_dict['refine'][0](logit_dict['refine'],
                                                      label_dict['alpha'])
            loss['refine'] = loss_refine
            loss_all = loss_all + loss_refine

        loss['all'] = loss_all
        return loss


class Refiner(nn.Layer):
    '''
    Refiner refines the coarse output to full resolution.

    Args:
        mode: area selection mode. Options:
            "full"         - No area selection, refine everywhere using regular Conv2d.
            "sampling"     - Refine fixed amount of pixels ranked by the top most errors.
            "thresholding" - Refine varying amount of pixels that have greater error than the threshold.
        sample_pixels: number of pixels to refine. Only used when mode == "sampling".
        threshold: error threshold ranged from 0 ~ 1. Refine where err > threshold. Only used when mode == "thresholding".
        kernel_size: The convolution kernel_size. Options: [1, 3]. Default: 3.
        prevent_oversampling: True for regular cases, False for speedtest.Default: True.
    '''

    def __init__(self,
                 mode,
                 sample_pixels,
                 threshold,
                 kernel_size=3,
                 prevent_oversampling=True):
        super().__init__()
        if mode not in ['full', 'sampling', 'thresholding']:
            raise ValueError(
                "mode must be in ['full', 'sampling', 'thresholding']")
        if kernel_size not in [1, 3]:
            raise ValueError("kernel_size must be in [1, 3]")

        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.prevent_oversampling = prevent_oversampling

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
        h_full, w_full = src.shape[2:]
        h_half, w_half = h_full // 2, w_full // 2
        h_quat, w_quat = h_full // 4, w_full // 4

        if self.mode != 'full':
            err = F.interpolate(
                err, (h_quat, w_quat), mode='bilinear', align_corners=False)
            ref = self.select_refinement_regions(err)
            idx = paddle.nonzero(ref.squeeze(1))

            if idx.shape[0] > 0:
                x = paddle.concat([hid, pha, tri], axis=1)
                x = F.interpolate(
                    x, (h_half, w_half), mode='bilinear', align_corners=False)
                x = self.crop_patch(x, idx, 2,
                                    3 if self.kernel_size == 3 else 0)

                y = F.interpolate(
                    src, (h_half, w_half), mode='bilinear', align_corners=False)
                y = self.crop_patch(y, idx, 2,
                                    3 if self.kernel_size == 3 else 0)

                x = self.conv1(paddle.concat([x, y], axis=1))
                x = self.conv2(x)

                x = F.interpolate(
                    x, (8, 8) if self.kernel_size == 3 else (4, 4),
                    mode='nearest')
                y = self.crop_patch(src, idx, 4,
                                    2 if self.kernel_size == 3 else 0)

                x = self.conv3(paddle.concat([x, y], axis=1))
                x = self.conv4(x)

                pha = F.interpolate(
                    pha, (h_full, w_full), mode='bilinear', align_corners=False)
                pha = self.replace_patch(pha, x, idx)
            else:
                pha = F.interpolate(
                    pha, (h_full, w_full), mode='bilinear', align_corners=False)

        else:
            x = paddle.concat([hid, pha, tri], axis=1)
            x = F.interpolate(
                x, (h_half, w_half), mode='bilinear', align_corners=False)
            y = F.interpolate(
                src, (h_half, w_half), mode='bilinear', align_corners=False)

            if self.kernel_size == 3:
                x = F.pad(x, [3, 3, 3, 3])
                y = F.pad(y, [3, 3, 3, 3])

            x = self.conv1(paddle.concat([x, y], axis=1))
            x = self.conv2(x)

            if self.kernel_size == 3:
                x = F.interpolate(x, (h_full + 4, w_full + 4))
                y = F.pad(src, [2, 2, 2, 2])
            else:
                x = F.interpolate(x, (h_full, w_full), mode='nearest')
                y = src

            x = self.conv3(paddle.concat([x, y], axis=1))
            x = self.conv4(x)

            pha = x
            ref = paddle.ones((src.shape[0], 1, h_quat, w_quat))
        return pha, ref

    def select_refinement_regions(self, err):
        '''
        select refinement regions.

        Args:
            err: error map (B, 1, H, W).

        Returns:
            Teosor: refinement regions (B, 1, H, W). 1 is selected, 0 is not.
        '''
        err.stop_gradient = True
        if self.mode == 'sampling':
            b, _, h, w = err.shape
            err = paddle.reshape(err, (b, -1))
            _, idx = err.topk(self.sample_pixels // 16, axis=1, sorted=False)
            ref = paddle.zeros_like(err)
            update = paddle.ones_like(idx, dtype='float32')
            for i in range(b):
                ref[i] = paddle.scatter(ref[i], idx[i], update[i])
            if self.prevent_oversampling:
                ref = ref * ((err > 0).astype('float32'))
            ref = ref.reshape((b, 1, h, w))
        else:
            ref = (err > self.threshold).astype('float32')
        return ref

    def crop_patch(self, x, idx, size, padding):
        """
        Crops selected patches from image given indices.

        Inputs:
            x: image (B, C, H, W).
            idx: selection indices shape is (p, 3), where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            patch: (P, C, h, w), where h = w = size + 2 * padding.
        """
        b, c, h, w = x.shape
        kernel_size = size + 2 * padding
        x = F.unfold(
            x, kernel_sizes=kernel_size, strides=size, paddings=padding)
        hout = int((h + 2 * padding - kernel_size) / size + 1)
        wout = int((w + 2 * padding - kernel_size) / size + 1)
        x = x.reshape((b, c, kernel_size, kernel_size, hout, wout))
        x = paddle.transpose(
            x, (0, 4, 5, 1, 2, 3)
        )  # If size is lager （4, 512, 512, 36, 8, 8), it will result OSError: (External)  Cuda error(700), an illegal memory access was encountered. idx will illgegal.
        patchs = paddle.gather_nd(x, idx)
        return patchs

    def replace_patch(self, x, y, idx):
        '''
        Replaces patches back into image given index.

        Args:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices shape is (p, 3), where the 3 values are (B, H, W) index.

        Returns:
            Tensor: (B, C, H, W), where patches at idx locations are replaced with y.
        '''
        bx, cx, hx, wx = x.shape
        by, cy, hy, wy = y.shape

        x = x.reshape((bx, cx, hx // hy, hy, wx // wy, wy))
        x = x.transpose((0, 2, 4, 1, 3, 5))
        ones = paddle.ones((idx.shape[0], cx, hy, wy))
        flag = paddle.scatter_nd(
            idx, ones, shape=x.shape)  # Get the index which should be replace
        x = x * (1 - flag)
        x = paddle.scatter_nd_add(x, idx, y)
        x = x.transpose((0, 3, 1, 4, 2, 5))
        x = x.reshape((bx, cx, hx, wx))
        return x


if __name__ == '__main__':
    #     paddle.set_device('cpu')
    import time
    from resnet_vd import ResNet34_vd
    backbone = ResNet34_vd(output_stride=32)
    x = paddle.randint(0, 256, (1, 3, 2048, 2048)).astype('float32')
    inputs = {}
    inputs['img'] = x

    model = ZiYanRefine(
        backbone=backbone,
        pretrained=None,
        backbone_scale=0.25,
        refine_mode='sampling',
        refine_sample_pixels=80000,
        refine_threshold=0.1,
        refine_kernel_size=3,
        refine_prevent_oversampling=True,
        if_refine=True)
    #     model.eval()
    for i in range(1):
        pha = model(inputs)
    print(pha)
#     for k, v in output.items():
#         print(k)
#         print(v)

#     refiner = Refiner(mode='sampling',
#                      sample_pixels=5000,
#                      threshold=0.1,
#                      kernel_size=3,
#                      prevent_oversampling=True)
# check select_refinement_regions, succeed
#     err = paddle.rand((2, 1, 512, 512))
#     start = time.time()
#     ref = refiner.select_refinement_regions_(err)
#     print('old time comsumn: ',time.time() - start)
#     print('old err')
#     print(err)
#     print('old ref')
#     print(ref)

# check crop_patch, succeed
#     x = paddle.rand((2, 3, 256, 256))
#     err = paddle.rand((2, 1, 128, 128))
#     ref = refiner.select_refinement_regions(err)
#     idx = paddle.nonzero(ref.squeeze(1))
#     idx = idx[:, 0], idx[:, 1], idx[:, 2]
#     size = 2
#     padding= 3
#     p = refiner.crop_patch(x, idx, size, padding)

# check replace_patch, succeed
#     p = p+1
#     p = p[:, :, 3:5, 3:5]
#     start = time.time()
#     refinement = refiner.replace_patch(x, p, idx)
#     print('replace_patch time:', time.time() - start)
#     print(refinement)

#     # check refine, succeed
#     src = paddle.rand((2, 3, 16, 16))
#     pha = paddle.rand((2, 1, 4, 4))
#     err = paddle.rand((2, 1, 4, 4))
#     hid = paddle.rand((2, 32, 4, 4))
#     tri = paddle.rand((2, 3, 4, 4))

#     pha_ref, ref = refiner(src, pha, err, hid, tri)
#     print('err')
#     print(err[1])
#     print('ref')
#     print(ref[1])
#     print('pha')
#     pha = F.interpolate(pha, (16, 16), mode='bilinear', align_corners=False)
#     print(pha[1,0,:,:])
#     print('pha_ref')
#     print(pha_ref[1,0,:,:])
#     print(pha_ref.shape, ref.shape)
