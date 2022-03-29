import math

import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils
from paddleseg.models import layers
from .ocrnet_nv import OCRNetNV


@manager.MODELS.add_component
class MscaleOCRNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 n_scales=[0.5, 1.0, 2.0],
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.ocrnet = OCRNetNV(
            num_classes,
            backbone,
            backbone_indices,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels,
            align_corners=align_corners,
            ms_attention=True)
        self.scale_attn = AttenHead(in_ch=ocr_mid_channels, out_ch=1)

        self.n_scales = n_scales
        self.pretrained = pretrained
        self.align_corners = align_corners

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward(self, x):
        if self.training:
            return self.two_scale_forward(x)
        else:
            return self.nscale_forward(x, self.n_scales)

    def two_scale_forward(self, x_1x):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        x_lo = nn.functional.interpolate(
            x_1x,
            scale_factor=0.5,
            align_corners=self.align_corners,
            mode='bilinear')
        lo_outs = self.single_scale_forward(x_lo)

        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']

        hi_outs = self.single_scale_forward(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = p_lo * logit_attn
        aux_lo = aux_lo * logit_attn
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + p_1x * (1 - logit_attn)
        joint_aux = aux_lo + aux_1x * (1 - logit_attn)

        output = [joint_pred, joint_aux]

        # Optionally, apply supervision to the multi-scale predictions
        # directly.
        scaled_pred_05x = scale_as(pred_05x, p_1x)
        output.extend([scaled_pred_05x, pred_10x])
        output.extend(output)
        return output

    def nscale_forward(self, x_1x, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        x_1x:
          scales - a list of scales to evaluate
          x_1x - dict containing 'images', the x_1x, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None

        for s in scales:
            x = nn.functional.interpolate(
                x_1x,
                scale_factor=s,
                align_corners=self.align_corners,
                mode='bilinear')
            outs = self.single_scale_forward(x)

            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']

            if pred is None:
                pred = cls_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out, self.align_corners)
                pred = cls_out * attn_out + pred * (1 - attn_out)
            else:
                # s < 1.0: upscale current
                cls_out = cls_out * attn_out

                cls_out = scale_as(cls_out, pred, self.align_corners)
                attn_out = scale_as(attn_out, pred, self.align_corners)

                pred = cls_out + pred * (1 - attn_out)

        return [pred]

    def single_scale_forward(self, x):
        x_size = paddle.shape(x)[2:]
        cls_out, aux_out, ocr_mid_feats = self.ocrnet(x)
        attn = self.scale_attn(ocr_mid_feats)

        cls_out = nn.functional.interpolate(
            cls_out,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)
        aux_out = nn.functional.interpolate(
            aux_out,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)
        attn = nn.functional.interpolate(
            attn,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return {'cls_out': cls_out, 'aux_out': aux_out, 'logit_attn': attn}


class AttenHead(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super(AttenHead, self).__init__()
        # bottleneck channels for seg and attn heads
        bot_ch = 256

        self.atten_head = nn.Sequential(
            layers.ConvBNReLU(
                in_ch, bot_ch, 3, padding=1, bias_attr=False),
            layers.ConvBNReLU(
                bot_ch, bot_ch, 3, padding=1, bias_attr=False),
            nn.Conv2D(
                bot_ch, out_ch, kernel_size=(1, 1), bias_attr=False),
            nn.Sigmoid())

    def forward(self, x):
        return self.atten_head(x)


def scale_as(x, y, align_corners=False):
    '''
    scale x to the same size as y
    '''
    y_shape = paddle.shape(y)
    x_scaled = nn.functional.interpolate(
        x, size=y_shape[2:], mode='bilinear', align_corners=align_corners)
    return x_scaled
