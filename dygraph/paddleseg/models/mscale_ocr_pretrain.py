import math

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils
from paddleseg.models import layers
'''add init_weight, dropout and syncbn'''


@manager.MODELS.add_component
class MscaleOCRPretrain(fluid.dygraph.Layer):
    def __init__(self, params, num_classes=19, pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.highresolutionnet0 = HighResolutionNet(params=params)
        self.ocr_block0 = OCR_block(params=params)
        self.scale_attn0 = Scale_attn(params=params)
        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward(self, x, n_scales=[0.5, 1.0, 2.0]):
        if self.training:
            return self.two_scale_forward(x)
        else:
            return self.nscale_forward(x, n_scales)

    def two_scale_forward(self, x_1x):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """

        align_corners = False
        x_lo = paddle.nn.functional.interpolate(
            x_1x,
            scale_factor=0.5,
            align_corners=align_corners,
            mode='bilinear')
        lo_outs = self.single_scale_forward(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self.single_scale_forward(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        output = [joint_pred, joint_aux]
        if self.training:

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            SUPERVISED_MSCALE_WT = 0.05
            if SUPERVISED_MSCALE_WT:  ## sota=0.05
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                output.extend([scaled_pred_05x, pred_10x])
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
        scales = sorted(scales, reverse=True)  # [2, 1, 0.5]

        pred = None
        aux = None
        output_dict = {}

        align_corners = False
        for s in scales:
            x = paddle.nn.functional.interpolate(
                x_1x,
                scale_factor=s,
                align_corners=align_corners,
                mode='bilinear')
            outs = self.single_scale_forward(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs[
                'aux_out']  # aux are only used in training, can be deleted

            #             output_dict[fmt_scale('pred', s)] = cls_out
            #             if s != 2.0:
            #                 output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out, align_corners)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out, align_corners)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred, align_corners)
                aux_out = scale_as(aux_out, pred, align_corners)
                attn_out = scale_as(attn_out, pred, align_corners)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux


#         output_dict['pred'] = pred
#         return output_dict
        return [pred]

    def single_scale_forward(self, x0):
        x0 = fluid.dygraph.base.to_variable(value=x0)
        x1 = x0.shape[2]
        x2 = x0.shape[3]
        x3 = self.highresolutionnet0(x0)
        x4, x5, x6 = self.ocr_block0(x3)
        x7 = self.scale_attn0(x4)
        x8 = [x1, x2]
        x9 = isinstance(x8, paddle.fluid.Variable)
        if x9:
            x8 = x8.numpy().tolist()
        assert None == None, 'The None must be None!'
        x12 = paddle.nn.functional.interpolate(
            x=x5,
            size=x8,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x13 = [x1, x2]
        x14 = isinstance(x13, paddle.fluid.Variable)
        if x14:
            x13 = x13.numpy().tolist()
        assert None == None, 'The None must be None!'
        x17 = paddle.nn.functional.interpolate(
            x=x6,
            size=x13,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x18 = [x1, x2]
        x19 = isinstance(x18, paddle.fluid.Variable)
        if x19:
            x18 = x18.numpy().tolist()
        assert None == None, 'The None must be None!'
        x22 = paddle.nn.functional.interpolate(
            x=x7,
            size=x18,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        #         x23 = [x17, x12, x22]
        #         return x23
        return {
            'cls_out': x17,  # x17 -> cls_out
            'aux_out': x12,  # x12 -> aux_out
            'logit_attn': x22
        }  # x22 -> attn


def scale_as(x, y, align_corners=False):
    '''
    scale x to the same size as y
    '''
    y_size = y.shape[2], y.shape[3]
    x_scaled = paddle.nn.functional.interpolate(
        x, size=y_size, mode='bilinear', align_corners=align_corners)
    return x_scaled


class Backbone_stage2_0_fuse_layers__1_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage2_0_fuse_layers__1_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__3_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__3_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__5_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__5_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage3_0_fuse_layers__5_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__5_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__5_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__5_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__7_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__7_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__9_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__9_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage3_1_fuse_layers__9_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__9_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__9_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__9_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__11_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__11_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__13_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__13_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage3_2_fuse_layers__13_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__13_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__13_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__13_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__15_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__15_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__17_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__17_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage3_3_fuse_layers__17_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__17_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__17_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__17_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__20_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__20_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__23_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__23_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_0_fuse_layers__23_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__23_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__23_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__23_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__25_3_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_0_fuse_layers__25_3_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_0_fuse_layers__25_3_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__25_3_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_0_fuse_layers__25_3_1_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_1_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__26_3_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__26_3_2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__29_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__29_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__32_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__32_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_1_fuse_layers__32_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__32_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__32_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__32_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__34_3_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_1_fuse_layers__34_3_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_1_fuse_layers__34_3_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__34_3_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_1_fuse_layers__34_3_1_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_1_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__35_3_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__35_3_2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__38_1_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__38_1_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__41_2_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__41_2_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_2_fuse_layers__41_2_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__41_2_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__41_2_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__41_2_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__43_3_0_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_2_fuse_layers__43_3_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_2_fuse_layers__43_3_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=48)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__43_3_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_stage4_2_fuse_layers__43_3_1_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_1_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__44_3_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__44_3_2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class BasicBlock(fluid.dygraph.Layer):
    def __init__(self, params, conv0_out_channels, conv0_in_channels,
                 bn0_num_channels, conv1_out_channels, conv1_in_channels,
                 bn1_num_channels):
        super(BasicBlock, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=conv0_out_channels,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=conv0_in_channels)
        self.bn0 = layers.SyncBatchNorm(
            bn0_num_channels, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            out_channels=conv1_out_channels,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=conv1_in_channels)
        self.bn1 = layers.SyncBatchNorm(
            bn1_num_channels, momentum=0.9, epsilon=1e-05)
        self.relu1 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = x5 + x0
        x7 = self.relu1(x6)
        return x7


class Backbone_stage4_2_fuse_layers__43_3_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_0, self).__init__()
        self.backbone_stage4_2_fuse_layers__43_3_0_00 = Backbone_stage4_2_fuse_layers__43_3_0_0(
            params=params)
        self.backbone_stage4_2_fuse_layers__43_3_0_10 = Backbone_stage4_2_fuse_layers__43_3_0_1(
            params=params)
        self.backbone_stage4_2_fuse_layers__43_3_0_20 = Backbone_stage4_2_fuse_layers__43_3_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__43_3_0_00(x0)
        x2 = self.backbone_stage4_2_fuse_layers__43_3_0_10(x1)
        x3 = self.backbone_stage4_2_fuse_layers__43_3_0_20(x2)
        return x3


class Backbone_stage3_0_fuse_layers__5_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__5_2_1, self).__init__()
        self.backbone_stage3_0_fuse_layers__5_2_1_00 = Backbone_stage3_0_fuse_layers__5_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__5_2_1_00(x0)
        return x1


class Backbone_stage3_1_fuse_layers__9_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__9_2_1, self).__init__()
        self.backbone_stage3_1_fuse_layers__9_2_1_00 = Backbone_stage3_1_fuse_layers__9_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__9_2_1_00(x0)
        return x1


class Backbone_stage3_0_fuse_layers__3_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__3_1_0, self).__init__()
        self.backbone_stage3_0_fuse_layers__3_1_0_00 = Backbone_stage3_0_fuse_layers__3_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__3_1_0_00(x0)
        return x1


class Backbone_stage4_0_fuse_layers__26_3_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__26_3_2, self).__init__()
        self.backbone_stage4_0_fuse_layers__26_3_2_00 = Backbone_stage4_0_fuse_layers__26_3_2_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__26_3_2_00(x0)
        return x1


class Backbone_stage4_2_fuse_layers__43_3_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__43_3_1, self).__init__()
        self.backbone_stage4_2_fuse_layers__43_3_1_00 = Backbone_stage4_2_fuse_layers__43_3_1_0(
            params=params)
        self.backbone_stage4_2_fuse_layers__43_3_1_10 = Backbone_stage4_2_fuse_layers__43_3_1_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__43_3_1_00(x0)
        x2 = self.backbone_stage4_2_fuse_layers__43_3_1_10(x1)
        return x2


class Backbone_stage4_2_fuse_layers__38_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__38_1_0, self).__init__()
        self.backbone_stage4_2_fuse_layers__38_1_0_00 = Backbone_stage4_2_fuse_layers__38_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__38_1_0_00(x0)
        return x1


class Backbone_stage4_1_fuse_layers__35_3_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__35_3_2, self).__init__()
        self.backbone_stage4_1_fuse_layers__35_3_2_00 = Backbone_stage4_1_fuse_layers__35_3_2_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__35_3_2_00(x0)
        return x1


class Backbone_stage3_3_fuse_layers__17_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__17_2_1, self).__init__()
        self.backbone_stage3_3_fuse_layers__17_2_1_00 = Backbone_stage3_3_fuse_layers__17_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__17_2_1_00(x0)
        return x1


class Backbone_stage3_1_fuse_layers__9_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__9_2_0, self).__init__()
        self.backbone_stage3_1_fuse_layers__9_2_0_00 = Backbone_stage3_1_fuse_layers__9_2_0_0(
            params=params)
        self.backbone_stage3_1_fuse_layers__9_2_0_10 = Backbone_stage3_1_fuse_layers__9_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__9_2_0_00(x0)
        x2 = self.backbone_stage3_1_fuse_layers__9_2_0_10(x1)
        return x2


class Backbone_stage4_1_fuse_layers__32_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__32_2_1, self).__init__()
        self.backbone_stage4_1_fuse_layers__32_2_1_00 = Backbone_stage4_1_fuse_layers__32_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__32_2_1_00(x0)
        return x1


class Backbone_stage2_0_fuse_layers__1_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage2_0_fuse_layers__1_1_0, self).__init__()
        self.backbone_stage2_0_fuse_layers__1_1_0_00 = Backbone_stage2_0_fuse_layers__1_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage2_0_fuse_layers__1_1_0_00(x0)
        return x1


class Backbone_stage4_0_fuse_layers__23_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__23_2_1, self).__init__()
        self.backbone_stage4_0_fuse_layers__23_2_1_00 = Backbone_stage4_0_fuse_layers__23_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__23_2_1_00(x0)
        return x1


class Backbone_stage4_0_fuse_layers__23_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__23_2_0, self).__init__()
        self.backbone_stage4_0_fuse_layers__23_2_0_00 = Backbone_stage4_0_fuse_layers__23_2_0_0(
            params=params)
        self.backbone_stage4_0_fuse_layers__23_2_0_10 = Backbone_stage4_0_fuse_layers__23_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__23_2_0_00(x0)
        x2 = self.backbone_stage4_0_fuse_layers__23_2_0_10(x1)
        return x2


class Backbone_stage4_0_fuse_layers__25_3_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_1, self).__init__()
        self.backbone_stage4_0_fuse_layers__25_3_1_00 = Backbone_stage4_0_fuse_layers__25_3_1_0(
            params=params)
        self.backbone_stage4_0_fuse_layers__25_3_1_10 = Backbone_stage4_0_fuse_layers__25_3_1_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__25_3_1_00(x0)
        x2 = self.backbone_stage4_0_fuse_layers__25_3_1_10(x1)
        return x2


class Backbone_stage3_2_fuse_layers__11_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__11_1_0, self).__init__()
        self.backbone_stage3_2_fuse_layers__11_1_0_00 = Backbone_stage3_2_fuse_layers__11_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__11_1_0_00(x0)
        return x1


class Backbone_stage3_3_fuse_layers__17_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__17_2_0, self).__init__()
        self.backbone_stage3_3_fuse_layers__17_2_0_00 = Backbone_stage3_3_fuse_layers__17_2_0_0(
            params=params)
        self.backbone_stage3_3_fuse_layers__17_2_0_10 = Backbone_stage3_3_fuse_layers__17_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__17_2_0_00(x0)
        x2 = self.backbone_stage3_3_fuse_layers__17_2_0_10(x1)
        return x2


class Backbone_stage4_1_fuse_layers__34_3_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_1, self).__init__()
        self.backbone_stage4_1_fuse_layers__34_3_1_00 = Backbone_stage4_1_fuse_layers__34_3_1_0(
            params=params)
        self.backbone_stage4_1_fuse_layers__34_3_1_10 = Backbone_stage4_1_fuse_layers__34_3_1_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__34_3_1_00(x0)
        x2 = self.backbone_stage4_1_fuse_layers__34_3_1_10(x1)
        return x2


class Backbone_stage4_1_fuse_layers__29_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__29_1_0, self).__init__()
        self.backbone_stage4_1_fuse_layers__29_1_0_00 = Backbone_stage4_1_fuse_layers__29_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__29_1_0_00(x0)
        return x1


class Backbone_stage3_0_fuse_layers__5_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__5_2_0, self).__init__()
        self.backbone_stage3_0_fuse_layers__5_2_0_00 = Backbone_stage3_0_fuse_layers__5_2_0_0(
            params=params)
        self.backbone_stage3_0_fuse_layers__5_2_0_10 = Backbone_stage3_0_fuse_layers__5_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__5_2_0_00(x0)
        x2 = self.backbone_stage3_0_fuse_layers__5_2_0_10(x1)
        return x2


class Backbone_stage4_1_fuse_layers__34_3_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__34_3_0, self).__init__()
        self.backbone_stage4_1_fuse_layers__34_3_0_00 = Backbone_stage4_1_fuse_layers__34_3_0_0(
            params=params)
        self.backbone_stage4_1_fuse_layers__34_3_0_10 = Backbone_stage4_1_fuse_layers__34_3_0_1(
            params=params)
        self.backbone_stage4_1_fuse_layers__34_3_0_20 = Backbone_stage4_1_fuse_layers__34_3_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__34_3_0_00(x0)
        x2 = self.backbone_stage4_1_fuse_layers__34_3_0_10(x1)
        x3 = self.backbone_stage4_1_fuse_layers__34_3_0_20(x2)
        return x3


class Backbone_stage4_2_fuse_layers__41_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__41_2_1, self).__init__()
        self.backbone_stage4_2_fuse_layers__41_2_1_00 = Backbone_stage4_2_fuse_layers__41_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__41_2_1_00(x0)
        return x1


class Backbone_stage3_2_fuse_layers__13_2_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__13_2_1, self).__init__()
        self.backbone_stage3_2_fuse_layers__13_2_1_00 = Backbone_stage3_2_fuse_layers__13_2_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__13_2_1_00(x0)
        return x1


class Backbone_stage3_3_fuse_layers__15_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__15_1_0, self).__init__()
        self.backbone_stage3_3_fuse_layers__15_1_0_00 = Backbone_stage3_3_fuse_layers__15_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__15_1_0_00(x0)
        return x1


class Backbone_stage4_0_fuse_layers__20_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__20_1_0, self).__init__()
        self.backbone_stage4_0_fuse_layers__20_1_0_00 = Backbone_stage4_0_fuse_layers__20_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__20_1_0_00(x0)
        return x1


class Backbone_stage4_1_fuse_layers__32_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__32_2_0, self).__init__()
        self.backbone_stage4_1_fuse_layers__32_2_0_00 = Backbone_stage4_1_fuse_layers__32_2_0_0(
            params=params)
        self.backbone_stage4_1_fuse_layers__32_2_0_10 = Backbone_stage4_1_fuse_layers__32_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__32_2_0_00(x0)
        x2 = self.backbone_stage4_1_fuse_layers__32_2_0_10(x1)
        return x2


class Backbone_stage4_2_fuse_layers__44_3_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__44_3_2, self).__init__()
        self.backbone_stage4_2_fuse_layers__44_3_2_00 = Backbone_stage4_2_fuse_layers__44_3_2_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__44_3_2_00(x0)
        return x1


class Backbone_stage3_1_fuse_layers__7_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__7_1_0, self).__init__()
        self.backbone_stage3_1_fuse_layers__7_1_0_00 = Backbone_stage3_1_fuse_layers__7_1_0_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__7_1_0_00(x0)
        return x1


class Backbone_stage4_2_fuse_layers__41_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__41_2_0, self).__init__()
        self.backbone_stage4_2_fuse_layers__41_2_0_00 = Backbone_stage4_2_fuse_layers__41_2_0_0(
            params=params)
        self.backbone_stage4_2_fuse_layers__41_2_0_10 = Backbone_stage4_2_fuse_layers__41_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__41_2_0_00(x0)
        x2 = self.backbone_stage4_2_fuse_layers__41_2_0_10(x1)
        return x2


class Backbone_stage3_2_fuse_layers__13_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__13_2_0, self).__init__()
        self.backbone_stage3_2_fuse_layers__13_2_0_00 = Backbone_stage3_2_fuse_layers__13_2_0_0(
            params=params)
        self.backbone_stage3_2_fuse_layers__13_2_0_10 = Backbone_stage3_2_fuse_layers__13_2_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__13_2_0_00(x0)
        x2 = self.backbone_stage3_2_fuse_layers__13_2_0_10(x1)
        return x2


class Backbone_stage4_0_fuse_layers__25_3_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__25_3_0, self).__init__()
        self.backbone_stage4_0_fuse_layers__25_3_0_00 = Backbone_stage4_0_fuse_layers__25_3_0_0(
            params=params)
        self.backbone_stage4_0_fuse_layers__25_3_0_10 = Backbone_stage4_0_fuse_layers__25_3_0_1(
            params=params)
        self.backbone_stage4_0_fuse_layers__25_3_0_20 = Backbone_stage4_0_fuse_layers__25_3_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__25_3_0_00(x0)
        x2 = self.backbone_stage4_0_fuse_layers__25_3_0_10(x1)
        x3 = self.backbone_stage4_0_fuse_layers__25_3_0_20(x2)
        return x3


class Backbone_stage2_0_fuse_layers_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage2_0_fuse_layers_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__1_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__1_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__2_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__2_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_0_fuse_layers__4_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_0_fuse_layers__4_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__5_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__5_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__6_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__6_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_1_fuse_layers__8_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_1_fuse_layers__8_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__9_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__9_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__10_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__10_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_2_fuse_layers__12_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_2_fuse_layers__12_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__13_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__13_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__14_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__14_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage3_3_fuse_layers__16_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3_3_fuse_layers__16_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__17_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__17_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__18_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__18_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__19_0_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__19_0_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__21_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__21_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__22_1_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__22_1_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_0_fuse_layers__24_2_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_0_fuse_layers__24_2_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__26_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__26_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__27_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__27_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__28_0_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__28_0_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__30_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__30_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__31_1_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__31_1_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_1_fuse_layers__33_2_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_1_fuse_layers__33_2_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__35_0_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__35_0_1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__36_0_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__36_0_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__37_0_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__37_0_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__39_1_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__39_1_2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__40_1_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__40_1_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage4_2_fuse_layers__42_2_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4_2_fuse_layers__42_2_3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=384)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_stage2_0_branches_0(fluid.dygraph.Layer):
    def __init__(self, params, basicblock0_conv0_out_channels,
                 basicblock0_conv0_in_channels, basicblock0_bn0_num_channels,
                 basicblock0_conv1_out_channels, basicblock0_conv1_in_channels,
                 basicblock0_bn1_num_channels, basicblock1_conv0_out_channels,
                 basicblock1_conv0_in_channels, basicblock1_bn0_num_channels,
                 basicblock1_conv1_out_channels, basicblock1_conv1_in_channels,
                 basicblock1_bn1_num_channels, basicblock2_conv0_out_channels,
                 basicblock2_conv0_in_channels, basicblock2_bn0_num_channels,
                 basicblock2_conv1_out_channels, basicblock2_conv1_in_channels,
                 basicblock2_bn1_num_channels, basicblock3_conv0_out_channels,
                 basicblock3_conv0_in_channels, basicblock3_bn0_num_channels,
                 basicblock3_conv1_out_channels, basicblock3_conv1_in_channels,
                 basicblock3_bn1_num_channels):
        super(Backbone_stage2_0_branches_0, self).__init__()
        self.basicblock0 = BasicBlock(
            conv0_out_channels=basicblock0_conv0_out_channels,
            conv0_in_channels=basicblock0_conv0_in_channels,
            bn0_num_channels=basicblock0_bn0_num_channels,
            conv1_out_channels=basicblock0_conv1_out_channels,
            conv1_in_channels=basicblock0_conv1_in_channels,
            bn1_num_channels=basicblock0_bn1_num_channels,
            params=params)
        self.basicblock1 = BasicBlock(
            conv0_out_channels=basicblock1_conv0_out_channels,
            conv0_in_channels=basicblock1_conv0_in_channels,
            bn0_num_channels=basicblock1_bn0_num_channels,
            conv1_out_channels=basicblock1_conv1_out_channels,
            conv1_in_channels=basicblock1_conv1_in_channels,
            bn1_num_channels=basicblock1_bn1_num_channels,
            params=params)
        self.basicblock2 = BasicBlock(
            conv0_out_channels=basicblock2_conv0_out_channels,
            conv0_in_channels=basicblock2_conv0_in_channels,
            bn0_num_channels=basicblock2_bn0_num_channels,
            conv1_out_channels=basicblock2_conv1_out_channels,
            conv1_in_channels=basicblock2_conv1_in_channels,
            bn1_num_channels=basicblock2_bn1_num_channels,
            params=params)
        self.basicblock3 = BasicBlock(
            conv0_out_channels=basicblock3_conv0_out_channels,
            conv0_in_channels=basicblock3_conv0_in_channels,
            bn0_num_channels=basicblock3_bn0_num_channels,
            conv1_out_channels=basicblock3_conv1_out_channels,
            conv1_in_channels=basicblock3_conv1_in_channels,
            bn1_num_channels=basicblock3_bn1_num_channels,
            params=params)

    def forward(self, x0):
        x1 = self.basicblock0(x0)
        x2 = self.basicblock1(x1)
        x3 = self.basicblock2(x2)
        x4 = self.basicblock3(x3)
        return x4


class Ocr_ocr_distri_head_object_context_block_f_pixel_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_pixel_1,
              self).__init__()
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_pixel_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_pixel_3,
              self).__init__()
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_object_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_object_1,
              self).__init__()
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_object_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_object_3,
              self).__init__()
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_down_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_down_1,
              self).__init__()
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_up_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_up_1, self).__init__()
        self.bn0 = layers.SyncBatchNorm(512, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class ModuleList52(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList52, self).__init__()
        self.backbone_stage2_0_fuse_layers_0_10 = Backbone_stage2_0_fuse_layers_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage2_0_fuse_layers_0_10(x0)
        return x1


class ModuleList53(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList53, self).__init__()
        self.backbone_stage2_0_fuse_layers__1_1_00 = Backbone_stage2_0_fuse_layers__1_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage2_0_fuse_layers__1_1_00(x0)
        return x1


class ModuleList54(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList54, self).__init__()
        self.backbone_stage3_0_fuse_layers__1_0_10 = Backbone_stage3_0_fuse_layers__1_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__1_0_10(x0)
        return x1


class ModuleList55(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList55, self).__init__()
        self.backbone_stage3_0_fuse_layers__2_0_20 = Backbone_stage3_0_fuse_layers__2_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__2_0_20(x0)
        return x1


class ModuleList56(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList56, self).__init__()
        self.backbone_stage3_0_fuse_layers__3_1_00 = Backbone_stage3_0_fuse_layers__3_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__3_1_00(x0)
        return x1


class ModuleList57(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList57, self).__init__()
        self.backbone_stage3_0_fuse_layers__4_1_20 = Backbone_stage3_0_fuse_layers__4_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_0_fuse_layers__4_1_20(x0)
        return x1


class ModuleList58(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList58, self).__init__()
        self.backbone_stage3_0_fuse_layers__5_2_00 = Backbone_stage3_0_fuse_layers__5_2_0(
            params=params)
        self.backbone_stage3_0_fuse_layers__5_2_10 = Backbone_stage3_0_fuse_layers__5_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage3_0_fuse_layers__5_2_00(x0)
        x3 = self.backbone_stage3_0_fuse_layers__5_2_10(x2)
        return x1, x3


class ModuleList59(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList59, self).__init__()
        self.backbone_stage3_1_fuse_layers__5_0_10 = Backbone_stage3_1_fuse_layers__5_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__5_0_10(x0)
        return x1


class ModuleList60(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList60, self).__init__()
        self.backbone_stage3_1_fuse_layers__6_0_20 = Backbone_stage3_1_fuse_layers__6_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__6_0_20(x0)
        return x1


class ModuleList61(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList61, self).__init__()
        self.backbone_stage3_1_fuse_layers__7_1_00 = Backbone_stage3_1_fuse_layers__7_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__7_1_00(x0)
        return x1


class ModuleList62(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList62, self).__init__()
        self.backbone_stage3_1_fuse_layers__8_1_20 = Backbone_stage3_1_fuse_layers__8_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_1_fuse_layers__8_1_20(x0)
        return x1


class ModuleList63(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList63, self).__init__()
        self.backbone_stage3_1_fuse_layers__9_2_00 = Backbone_stage3_1_fuse_layers__9_2_0(
            params=params)
        self.backbone_stage3_1_fuse_layers__9_2_10 = Backbone_stage3_1_fuse_layers__9_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage3_1_fuse_layers__9_2_00(x0)
        x3 = self.backbone_stage3_1_fuse_layers__9_2_10(x2)
        return x1, x3


class ModuleList64(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList64, self).__init__()
        self.backbone_stage3_2_fuse_layers__9_0_10 = Backbone_stage3_2_fuse_layers__9_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__9_0_10(x0)
        return x1


class ModuleList65(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList65, self).__init__()
        self.backbone_stage3_2_fuse_layers__10_0_20 = Backbone_stage3_2_fuse_layers__10_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__10_0_20(x0)
        return x1


class ModuleList66(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList66, self).__init__()
        self.backbone_stage3_2_fuse_layers__11_1_00 = Backbone_stage3_2_fuse_layers__11_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__11_1_00(x0)
        return x1


class ModuleList67(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList67, self).__init__()
        self.backbone_stage3_2_fuse_layers__12_1_20 = Backbone_stage3_2_fuse_layers__12_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_2_fuse_layers__12_1_20(x0)
        return x1


class ModuleList68(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList68, self).__init__()
        self.backbone_stage3_2_fuse_layers__13_2_00 = Backbone_stage3_2_fuse_layers__13_2_0(
            params=params)
        self.backbone_stage3_2_fuse_layers__13_2_10 = Backbone_stage3_2_fuse_layers__13_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage3_2_fuse_layers__13_2_00(x0)
        x3 = self.backbone_stage3_2_fuse_layers__13_2_10(x2)
        return x1, x3


class ModuleList69(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList69, self).__init__()
        self.backbone_stage3_3_fuse_layers__13_0_10 = Backbone_stage3_3_fuse_layers__13_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__13_0_10(x0)
        return x1


class ModuleList70(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList70, self).__init__()
        self.backbone_stage3_3_fuse_layers__14_0_20 = Backbone_stage3_3_fuse_layers__14_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__14_0_20(x0)
        return x1


class ModuleList71(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList71, self).__init__()
        self.backbone_stage3_3_fuse_layers__15_1_00 = Backbone_stage3_3_fuse_layers__15_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__15_1_00(x0)
        return x1


class ModuleList72(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList72, self).__init__()
        self.backbone_stage3_3_fuse_layers__16_1_20 = Backbone_stage3_3_fuse_layers__16_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage3_3_fuse_layers__16_1_20(x0)
        return x1


class ModuleList73(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList73, self).__init__()
        self.backbone_stage3_3_fuse_layers__17_2_00 = Backbone_stage3_3_fuse_layers__17_2_0(
            params=params)
        self.backbone_stage3_3_fuse_layers__17_2_10 = Backbone_stage3_3_fuse_layers__17_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage3_3_fuse_layers__17_2_00(x0)
        x3 = self.backbone_stage3_3_fuse_layers__17_2_10(x2)
        return x1, x3


class ModuleList74(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList74, self).__init__()
        self.backbone_stage4_0_fuse_layers__17_0_10 = Backbone_stage4_0_fuse_layers__17_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__17_0_10(x0)
        return x1


class ModuleList75(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList75, self).__init__()
        self.backbone_stage4_0_fuse_layers__18_0_20 = Backbone_stage4_0_fuse_layers__18_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__18_0_20(x0)
        return x1


class ModuleList76(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList76, self).__init__()
        self.backbone_stage4_0_fuse_layers__19_0_30 = Backbone_stage4_0_fuse_layers__19_0_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__19_0_30(x0)
        return x1


class ModuleList77(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList77, self).__init__()
        self.backbone_stage4_0_fuse_layers__20_1_00 = Backbone_stage4_0_fuse_layers__20_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__20_1_00(x0)
        return x1


class ModuleList78(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList78, self).__init__()
        self.backbone_stage4_0_fuse_layers__21_1_20 = Backbone_stage4_0_fuse_layers__21_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__21_1_20(x0)
        return x1


class ModuleList79(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList79, self).__init__()
        self.backbone_stage4_0_fuse_layers__22_1_30 = Backbone_stage4_0_fuse_layers__22_1_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__22_1_30(x0)
        return x1


class ModuleList80(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList80, self).__init__()
        self.backbone_stage4_0_fuse_layers__23_2_00 = Backbone_stage4_0_fuse_layers__23_2_0(
            params=params)
        self.backbone_stage4_0_fuse_layers__23_2_10 = Backbone_stage4_0_fuse_layers__23_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_0_fuse_layers__23_2_00(x0)
        x3 = self.backbone_stage4_0_fuse_layers__23_2_10(x2)
        return x1, x3


class ModuleList81(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList81, self).__init__()
        self.backbone_stage4_0_fuse_layers__24_2_30 = Backbone_stage4_0_fuse_layers__24_2_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__24_2_30(x0)
        return x1


class ModuleList82(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList82, self).__init__()
        self.backbone_stage4_0_fuse_layers__25_3_00 = Backbone_stage4_0_fuse_layers__25_3_0(
            params=params)
        self.backbone_stage4_0_fuse_layers__25_3_10 = Backbone_stage4_0_fuse_layers__25_3_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_0_fuse_layers__25_3_00(x0)
        x3 = self.backbone_stage4_0_fuse_layers__25_3_10(x2)
        return x1, x3


class ModuleList83(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList83, self).__init__()
        self.backbone_stage4_0_fuse_layers__26_3_20 = Backbone_stage4_0_fuse_layers__26_3_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_0_fuse_layers__26_3_20(x0)
        return x1


class ModuleList84(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList84, self).__init__()
        self.backbone_stage4_1_fuse_layers__26_0_10 = Backbone_stage4_1_fuse_layers__26_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__26_0_10(x0)
        return x1


class ModuleList85(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList85, self).__init__()
        self.backbone_stage4_1_fuse_layers__27_0_20 = Backbone_stage4_1_fuse_layers__27_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__27_0_20(x0)
        return x1


class ModuleList86(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList86, self).__init__()
        self.backbone_stage4_1_fuse_layers__28_0_30 = Backbone_stage4_1_fuse_layers__28_0_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__28_0_30(x0)
        return x1


class ModuleList87(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList87, self).__init__()
        self.backbone_stage4_1_fuse_layers__29_1_00 = Backbone_stage4_1_fuse_layers__29_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__29_1_00(x0)
        return x1


class ModuleList88(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList88, self).__init__()
        self.backbone_stage4_1_fuse_layers__30_1_20 = Backbone_stage4_1_fuse_layers__30_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__30_1_20(x0)
        return x1


class ModuleList89(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList89, self).__init__()
        self.backbone_stage4_1_fuse_layers__31_1_30 = Backbone_stage4_1_fuse_layers__31_1_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__31_1_30(x0)
        return x1


class ModuleList90(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList90, self).__init__()
        self.backbone_stage4_1_fuse_layers__32_2_00 = Backbone_stage4_1_fuse_layers__32_2_0(
            params=params)
        self.backbone_stage4_1_fuse_layers__32_2_10 = Backbone_stage4_1_fuse_layers__32_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_1_fuse_layers__32_2_00(x0)
        x3 = self.backbone_stage4_1_fuse_layers__32_2_10(x2)
        return x1, x3


class ModuleList91(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList91, self).__init__()
        self.backbone_stage4_1_fuse_layers__33_2_30 = Backbone_stage4_1_fuse_layers__33_2_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__33_2_30(x0)
        return x1


class ModuleList92(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList92, self).__init__()
        self.backbone_stage4_1_fuse_layers__34_3_00 = Backbone_stage4_1_fuse_layers__34_3_0(
            params=params)
        self.backbone_stage4_1_fuse_layers__34_3_10 = Backbone_stage4_1_fuse_layers__34_3_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_1_fuse_layers__34_3_00(x0)
        x3 = self.backbone_stage4_1_fuse_layers__34_3_10(x2)
        return x1, x3


class ModuleList93(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList93, self).__init__()
        self.backbone_stage4_1_fuse_layers__35_3_20 = Backbone_stage4_1_fuse_layers__35_3_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_1_fuse_layers__35_3_20(x0)
        return x1


class ModuleList94(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList94, self).__init__()
        self.backbone_stage4_2_fuse_layers__35_0_10 = Backbone_stage4_2_fuse_layers__35_0_1(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__35_0_10(x0)
        return x1


class ModuleList95(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList95, self).__init__()
        self.backbone_stage4_2_fuse_layers__36_0_20 = Backbone_stage4_2_fuse_layers__36_0_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__36_0_20(x0)
        return x1


class ModuleList96(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList96, self).__init__()
        self.backbone_stage4_2_fuse_layers__37_0_30 = Backbone_stage4_2_fuse_layers__37_0_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__37_0_30(x0)
        return x1


class ModuleList97(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList97, self).__init__()
        self.backbone_stage4_2_fuse_layers__38_1_00 = Backbone_stage4_2_fuse_layers__38_1_0(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__38_1_00(x0)
        return x1


class ModuleList98(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList98, self).__init__()
        self.backbone_stage4_2_fuse_layers__39_1_20 = Backbone_stage4_2_fuse_layers__39_1_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__39_1_20(x0)
        return x1


class ModuleList99(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList99, self).__init__()
        self.backbone_stage4_2_fuse_layers__40_1_30 = Backbone_stage4_2_fuse_layers__40_1_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__40_1_30(x0)
        return x1


class ModuleList100(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList100, self).__init__()
        self.backbone_stage4_2_fuse_layers__41_2_00 = Backbone_stage4_2_fuse_layers__41_2_0(
            params=params)
        self.backbone_stage4_2_fuse_layers__41_2_10 = Backbone_stage4_2_fuse_layers__41_2_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_2_fuse_layers__41_2_00(x0)
        x3 = self.backbone_stage4_2_fuse_layers__41_2_10(x2)
        return x1, x3


class ModuleList101(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList101, self).__init__()
        self.backbone_stage4_2_fuse_layers__42_2_30 = Backbone_stage4_2_fuse_layers__42_2_3(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__42_2_30(x0)
        return x1


class ModuleList102(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList102, self).__init__()
        self.backbone_stage4_2_fuse_layers__43_3_00 = Backbone_stage4_2_fuse_layers__43_3_0(
            params=params)
        self.backbone_stage4_2_fuse_layers__43_3_10 = Backbone_stage4_2_fuse_layers__43_3_1(
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage4_2_fuse_layers__43_3_00(x0)
        x3 = self.backbone_stage4_2_fuse_layers__43_3_10(x2)
        return x1, x3


class ModuleList103(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList103, self).__init__()
        self.backbone_stage4_2_fuse_layers__44_3_20 = Backbone_stage4_2_fuse_layers__44_3_2(
            params=params)

    def forward(self, x0):
        x1 = self.backbone_stage4_2_fuse_layers__44_3_20(x0)
        return x1


class Ocr_ocr_distri_head_object_context_block_f_down(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_down, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=512)
        self.ocr_ocr_distri_head_object_context_block_f_down_10 = Ocr_ocr_distri_head_object_context_block_f_down_1(
            params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_ocr_distri_head_object_context_block_f_down_10(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_pixel(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_pixel, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=512)
        self.ocr_ocr_distri_head_object_context_block_f_pixel_10 = Ocr_ocr_distri_head_object_context_block_f_pixel_1(
            params=params)
        self.conv1 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.ocr_ocr_distri_head_object_context_block_f_pixel_30 = Ocr_ocr_distri_head_object_context_block_f_pixel_3(
            params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_ocr_distri_head_object_context_block_f_pixel_10(x1)
        x3 = self.conv1(x2)
        x4 = self.ocr_ocr_distri_head_object_context_block_f_pixel_30(x3)
        return x4


class Ocr_ocr_distri_head_object_context_block_f_up(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_up, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=512,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.ocr_ocr_distri_head_object_context_block_f_up_10 = Ocr_ocr_distri_head_object_context_block_f_up_1(
            params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_ocr_distri_head_object_context_block_f_up_10(x1)
        return x2


class Ocr_ocr_distri_head_object_context_block_f_object(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_object_context_block_f_object,
              self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=512)
        self.ocr_ocr_distri_head_object_context_block_f_object_10 = Ocr_ocr_distri_head_object_context_block_f_object_1(
            params=params)
        self.conv1 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.ocr_ocr_distri_head_object_context_block_f_object_30 = Ocr_ocr_distri_head_object_context_block_f_object_3(
            params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_ocr_distri_head_object_context_block_f_object_10(x1)
        x3 = self.conv1(x2)
        x4 = self.ocr_ocr_distri_head_object_context_block_f_object_30(x3)
        return x4


class Backbone_layer1_0_downsample(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_layer1_0_downsample, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Backbone_transition1_1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition1_1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=96,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.bn0 = layers.SyncBatchNorm(96, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_transition2_2_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition2_2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=192,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=96)
        self.bn0 = layers.SyncBatchNorm(192, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Backbone_transition3_3_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition3_3_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=384,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=192)
        self.bn0 = layers.SyncBatchNorm(384, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Ocr_ocr_distri_head_conv_bn_dropout_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_conv_bn_dropout_1, self).__init__()
        self.bn0 = layers.SyncBatchNorm(512, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.dropout0 = paddle.nn.Dropout2D(
            0.05)  ######################### add dropout

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        x3 = self.dropout0(x2)
        return x3


class ModuleList55_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList55_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)

    def forward(self, x0, x2):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        return x1, x3


class ModuleList56_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList56_0, self).__init__()
        self.modulelist520 = ModuleList52(params=params)

    def forward(self, x0):
        x1 = self.modulelist520(x0)
        return x1


class ModuleList57_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList57_0, self).__init__()
        self.modulelist530 = ModuleList53(params=params)

    def forward(self, x0):
        x1 = self.modulelist530(x0)
        return x1


class ModuleList58_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList58_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)

    def forward(self, x0, x2, x4):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        return x1, x3, x5


class ModuleList59_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList59_0, self).__init__()
        self.modulelist540 = ModuleList54(params=params)

    def forward(self, x0):
        x1 = self.modulelist540(x0)
        return x1


class ModuleList60_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList60_0, self).__init__()
        self.modulelist550 = ModuleList55(params=params)

    def forward(self, x0):
        x1 = self.modulelist550(x0)
        return x1


class ModuleList61_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList61_0, self).__init__()
        self.modulelist560 = ModuleList56(params=params)

    def forward(self, x0):
        x1 = self.modulelist560(x0)
        return x1


class ModuleList62_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList62_0, self).__init__()
        self.modulelist570 = ModuleList57(params=params)

    def forward(self, x0):
        x1 = self.modulelist570(x0)
        return x1


class ModuleList63_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList63_0, self).__init__()
        self.modulelist580 = ModuleList58(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist580(x0, x1)
        return x2, x3


class ModuleList64_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList64_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)

    def forward(self, x0, x2, x4):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        return x1, x3, x5


class ModuleList65_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList65_0, self).__init__()
        self.modulelist590 = ModuleList59(params=params)

    def forward(self, x0):
        x1 = self.modulelist590(x0)
        return x1


class ModuleList66_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList66_0, self).__init__()
        self.modulelist600 = ModuleList60(params=params)

    def forward(self, x0):
        x1 = self.modulelist600(x0)
        return x1


class ModuleList67_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList67_0, self).__init__()
        self.modulelist610 = ModuleList61(params=params)

    def forward(self, x0):
        x1 = self.modulelist610(x0)
        return x1


class ModuleList68_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList68_0, self).__init__()
        self.modulelist620 = ModuleList62(params=params)

    def forward(self, x0):
        x1 = self.modulelist620(x0)
        return x1


class ModuleList69_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList69_0, self).__init__()
        self.modulelist630 = ModuleList63(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist630(x0, x1)
        return x2, x3


class ModuleList70_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList70_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)

    def forward(self, x0, x2, x4):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        return x1, x3, x5


class ModuleList71_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList71_0, self).__init__()
        self.modulelist640 = ModuleList64(params=params)

    def forward(self, x0):
        x1 = self.modulelist640(x0)
        return x1


class ModuleList72_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList72_0, self).__init__()
        self.modulelist650 = ModuleList65(params=params)

    def forward(self, x0):
        x1 = self.modulelist650(x0)
        return x1


class ModuleList73_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList73_0, self).__init__()
        self.modulelist660 = ModuleList66(params=params)

    def forward(self, x0):
        x1 = self.modulelist660(x0)
        return x1


class ModuleList74_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList74_0, self).__init__()
        self.modulelist670 = ModuleList67(params=params)

    def forward(self, x0):
        x1 = self.modulelist670(x0)
        return x1


class ModuleList75_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList75_0, self).__init__()
        self.modulelist680 = ModuleList68(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist680(x0, x1)
        return x2, x3


class ModuleList76_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList76_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)

    def forward(self, x0, x2, x4):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        return x1, x3, x5


class ModuleList77_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList77_0, self).__init__()
        self.modulelist690 = ModuleList69(params=params)

    def forward(self, x0):
        x1 = self.modulelist690(x0)
        return x1


class ModuleList78_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList78_0, self).__init__()
        self.modulelist700 = ModuleList70(params=params)

    def forward(self, x0):
        x1 = self.modulelist700(x0)
        return x1


class ModuleList79_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList79_0, self).__init__()
        self.modulelist710 = ModuleList71(params=params)

    def forward(self, x0):
        x1 = self.modulelist710(x0)
        return x1


class ModuleList80_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList80_0, self).__init__()
        self.modulelist720 = ModuleList72(params=params)

    def forward(self, x0):
        x1 = self.modulelist720(x0)
        return x1


class ModuleList81_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList81_0, self).__init__()
        self.modulelist730 = ModuleList73(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist730(x0, x1)
        return x2, x3


class ModuleList82_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList82_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)
        self.backbone_stage2_0_branches_03 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=384,
            basicblock0_conv0_in_channels=384,
            basicblock0_bn0_num_channels=384,
            basicblock0_conv1_out_channels=384,
            basicblock0_conv1_in_channels=384,
            basicblock0_bn1_num_channels=384,
            basicblock1_conv0_out_channels=384,
            basicblock1_conv0_in_channels=384,
            basicblock1_bn0_num_channels=384,
            basicblock1_conv1_out_channels=384,
            basicblock1_conv1_in_channels=384,
            basicblock1_bn1_num_channels=384,
            basicblock2_conv0_out_channels=384,
            basicblock2_conv0_in_channels=384,
            basicblock2_bn0_num_channels=384,
            basicblock2_conv1_out_channels=384,
            basicblock2_conv1_in_channels=384,
            basicblock2_bn1_num_channels=384,
            basicblock3_conv0_out_channels=384,
            basicblock3_conv0_in_channels=384,
            basicblock3_bn0_num_channels=384,
            basicblock3_conv1_out_channels=384,
            basicblock3_conv1_in_channels=384,
            basicblock3_bn1_num_channels=384,
            params=params)

    def forward(self, x0, x2, x4, x6):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        x7 = self.backbone_stage2_0_branches_03(x6)
        return x1, x3, x5, x7


class ModuleList83_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList83_0, self).__init__()
        self.modulelist740 = ModuleList74(params=params)

    def forward(self, x0):
        x1 = self.modulelist740(x0)
        return x1


class ModuleList84_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList84_0, self).__init__()
        self.modulelist750 = ModuleList75(params=params)

    def forward(self, x0):
        x1 = self.modulelist750(x0)
        return x1


class ModuleList85_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList85_0, self).__init__()
        self.modulelist760 = ModuleList76(params=params)

    def forward(self, x0):
        x1 = self.modulelist760(x0)
        return x1


class ModuleList86_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList86_0, self).__init__()
        self.modulelist770 = ModuleList77(params=params)

    def forward(self, x0):
        x1 = self.modulelist770(x0)
        return x1


class ModuleList87_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList87_0, self).__init__()
        self.modulelist780 = ModuleList78(params=params)

    def forward(self, x0):
        x1 = self.modulelist780(x0)
        return x1


class ModuleList88_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList88_0, self).__init__()
        self.modulelist790 = ModuleList79(params=params)

    def forward(self, x0):
        x1 = self.modulelist790(x0)
        return x1


class ModuleList89_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList89_0, self).__init__()
        self.modulelist800 = ModuleList80(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist800(x0, x1)
        return x2, x3


class ModuleList90_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList90_0, self).__init__()
        self.modulelist810 = ModuleList81(params=params)

    def forward(self, x0):
        x1 = self.modulelist810(x0)
        return x1


class ModuleList91_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList91_0, self).__init__()
        self.modulelist820 = ModuleList82(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist820(x0, x1)
        return x2, x3


class ModuleList92_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList92_0, self).__init__()
        self.modulelist830 = ModuleList83(params=params)

    def forward(self, x0):
        x1 = self.modulelist830(x0)
        return x1


class ModuleList93_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList93_0, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)
        self.backbone_stage2_0_branches_03 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=384,
            basicblock0_conv0_in_channels=384,
            basicblock0_bn0_num_channels=384,
            basicblock0_conv1_out_channels=384,
            basicblock0_conv1_in_channels=384,
            basicblock0_bn1_num_channels=384,
            basicblock1_conv0_out_channels=384,
            basicblock1_conv0_in_channels=384,
            basicblock1_bn0_num_channels=384,
            basicblock1_conv1_out_channels=384,
            basicblock1_conv1_in_channels=384,
            basicblock1_bn1_num_channels=384,
            basicblock2_conv0_out_channels=384,
            basicblock2_conv0_in_channels=384,
            basicblock2_bn0_num_channels=384,
            basicblock2_conv1_out_channels=384,
            basicblock2_conv1_in_channels=384,
            basicblock2_bn1_num_channels=384,
            basicblock3_conv0_out_channels=384,
            basicblock3_conv0_in_channels=384,
            basicblock3_bn0_num_channels=384,
            basicblock3_conv1_out_channels=384,
            basicblock3_conv1_in_channels=384,
            basicblock3_bn1_num_channels=384,
            params=params)

    def forward(self, x0, x2, x4, x6):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        x7 = self.backbone_stage2_0_branches_03(x6)
        return x1, x3, x5, x7


class ModuleList94_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList94_0, self).__init__()
        self.modulelist840 = ModuleList84(params=params)

    def forward(self, x0):
        x1 = self.modulelist840(x0)
        return x1


class ModuleList95_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList95_0, self).__init__()
        self.modulelist850 = ModuleList85(params=params)

    def forward(self, x0):
        x1 = self.modulelist850(x0)
        return x1


class ModuleList96_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList96_0, self).__init__()
        self.modulelist860 = ModuleList86(params=params)

    def forward(self, x0):
        x1 = self.modulelist860(x0)
        return x1


class ModuleList97_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList97_0, self).__init__()
        self.modulelist870 = ModuleList87(params=params)

    def forward(self, x0):
        x1 = self.modulelist870(x0)
        return x1


class ModuleList98_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList98_0, self).__init__()
        self.modulelist880 = ModuleList88(params=params)

    def forward(self, x0):
        x1 = self.modulelist880(x0)
        return x1


class ModuleList99_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList99_0, self).__init__()
        self.modulelist890 = ModuleList89(params=params)

    def forward(self, x0):
        x1 = self.modulelist890(x0)
        return x1


class ModuleList100_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList100_0, self).__init__()
        self.modulelist900 = ModuleList90(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist900(x0, x1)
        return x2, x3


class ModuleList101_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList101_0, self).__init__()
        self.modulelist910 = ModuleList91(params=params)

    def forward(self, x0):
        x1 = self.modulelist910(x0)
        return x1


class ModuleList102_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList102_0, self).__init__()
        self.modulelist920 = ModuleList92(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist920(x0, x1)
        return x2, x3


class ModuleList103_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList103_0, self).__init__()
        self.modulelist930 = ModuleList93(params=params)

    def forward(self, x0):
        x1 = self.modulelist930(x0)
        return x1


class ModuleList104(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList104, self).__init__()
        self.backbone_stage2_0_branches_00 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=48,
            basicblock0_conv0_in_channels=48,
            basicblock0_bn0_num_channels=48,
            basicblock0_conv1_out_channels=48,
            basicblock0_conv1_in_channels=48,
            basicblock0_bn1_num_channels=48,
            basicblock1_conv0_out_channels=48,
            basicblock1_conv0_in_channels=48,
            basicblock1_bn0_num_channels=48,
            basicblock1_conv1_out_channels=48,
            basicblock1_conv1_in_channels=48,
            basicblock1_bn1_num_channels=48,
            basicblock2_conv0_out_channels=48,
            basicblock2_conv0_in_channels=48,
            basicblock2_bn0_num_channels=48,
            basicblock2_conv1_out_channels=48,
            basicblock2_conv1_in_channels=48,
            basicblock2_bn1_num_channels=48,
            basicblock3_conv0_out_channels=48,
            basicblock3_conv0_in_channels=48,
            basicblock3_bn0_num_channels=48,
            basicblock3_conv1_out_channels=48,
            basicblock3_conv1_in_channels=48,
            basicblock3_bn1_num_channels=48,
            params=params)
        self.backbone_stage2_0_branches_01 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=96,
            basicblock0_conv0_in_channels=96,
            basicblock0_bn0_num_channels=96,
            basicblock0_conv1_out_channels=96,
            basicblock0_conv1_in_channels=96,
            basicblock0_bn1_num_channels=96,
            basicblock1_conv0_out_channels=96,
            basicblock1_conv0_in_channels=96,
            basicblock1_bn0_num_channels=96,
            basicblock1_conv1_out_channels=96,
            basicblock1_conv1_in_channels=96,
            basicblock1_bn1_num_channels=96,
            basicblock2_conv0_out_channels=96,
            basicblock2_conv0_in_channels=96,
            basicblock2_bn0_num_channels=96,
            basicblock2_conv1_out_channels=96,
            basicblock2_conv1_in_channels=96,
            basicblock2_bn1_num_channels=96,
            basicblock3_conv0_out_channels=96,
            basicblock3_conv0_in_channels=96,
            basicblock3_bn0_num_channels=96,
            basicblock3_conv1_out_channels=96,
            basicblock3_conv1_in_channels=96,
            basicblock3_bn1_num_channels=96,
            params=params)
        self.backbone_stage2_0_branches_02 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=192,
            basicblock0_conv0_in_channels=192,
            basicblock0_bn0_num_channels=192,
            basicblock0_conv1_out_channels=192,
            basicblock0_conv1_in_channels=192,
            basicblock0_bn1_num_channels=192,
            basicblock1_conv0_out_channels=192,
            basicblock1_conv0_in_channels=192,
            basicblock1_bn0_num_channels=192,
            basicblock1_conv1_out_channels=192,
            basicblock1_conv1_in_channels=192,
            basicblock1_bn1_num_channels=192,
            basicblock2_conv0_out_channels=192,
            basicblock2_conv0_in_channels=192,
            basicblock2_bn0_num_channels=192,
            basicblock2_conv1_out_channels=192,
            basicblock2_conv1_in_channels=192,
            basicblock2_bn1_num_channels=192,
            basicblock3_conv0_out_channels=192,
            basicblock3_conv0_in_channels=192,
            basicblock3_bn0_num_channels=192,
            basicblock3_conv1_out_channels=192,
            basicblock3_conv1_in_channels=192,
            basicblock3_bn1_num_channels=192,
            params=params)
        self.backbone_stage2_0_branches_03 = Backbone_stage2_0_branches_0(
            basicblock0_conv0_out_channels=384,
            basicblock0_conv0_in_channels=384,
            basicblock0_bn0_num_channels=384,
            basicblock0_conv1_out_channels=384,
            basicblock0_conv1_in_channels=384,
            basicblock0_bn1_num_channels=384,
            basicblock1_conv0_out_channels=384,
            basicblock1_conv0_in_channels=384,
            basicblock1_bn0_num_channels=384,
            basicblock1_conv1_out_channels=384,
            basicblock1_conv1_in_channels=384,
            basicblock1_bn1_num_channels=384,
            basicblock2_conv0_out_channels=384,
            basicblock2_conv0_in_channels=384,
            basicblock2_bn0_num_channels=384,
            basicblock2_conv1_out_channels=384,
            basicblock2_conv1_in_channels=384,
            basicblock2_bn1_num_channels=384,
            basicblock3_conv0_out_channels=384,
            basicblock3_conv0_in_channels=384,
            basicblock3_bn0_num_channels=384,
            basicblock3_conv1_out_channels=384,
            basicblock3_conv1_in_channels=384,
            basicblock3_bn1_num_channels=384,
            params=params)

    def forward(self, x0, x2, x4, x6):
        x1 = self.backbone_stage2_0_branches_00(x0)
        x3 = self.backbone_stage2_0_branches_01(x2)
        x5 = self.backbone_stage2_0_branches_02(x4)
        x7 = self.backbone_stage2_0_branches_03(x6)
        return x1, x3, x5, x7


class ModuleList105(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList105, self).__init__()
        self.modulelist940 = ModuleList94(params=params)

    def forward(self, x0):
        x1 = self.modulelist940(x0)
        return x1


class ModuleList106(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList106, self).__init__()
        self.modulelist950 = ModuleList95(params=params)

    def forward(self, x0):
        x1 = self.modulelist950(x0)
        return x1


class ModuleList107(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList107, self).__init__()
        self.modulelist960 = ModuleList96(params=params)

    def forward(self, x0):
        x1 = self.modulelist960(x0)
        return x1


class ModuleList108(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList108, self).__init__()
        self.modulelist970 = ModuleList97(params=params)

    def forward(self, x0):
        x1 = self.modulelist970(x0)
        return x1


class ModuleList109(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList109, self).__init__()
        self.modulelist980 = ModuleList98(params=params)

    def forward(self, x0):
        x1 = self.modulelist980(x0)
        return x1


class ModuleList110(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList110, self).__init__()
        self.modulelist990 = ModuleList99(params=params)

    def forward(self, x0):
        x1 = self.modulelist990(x0)
        return x1


class ModuleList111(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList111, self).__init__()
        self.modulelist1000 = ModuleList100(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist1000(x0, x1)
        return x2, x3


class ModuleList112(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList112, self).__init__()
        self.modulelist1010 = ModuleList101(params=params)

    def forward(self, x0):
        x1 = self.modulelist1010(x0)
        return x1


class ModuleList113(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList113, self).__init__()
        self.modulelist1020 = ModuleList102(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.modulelist1020(x0, x1)
        return x2, x3


class ModuleList114(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList114, self).__init__()
        self.modulelist1030 = ModuleList103(params=params)

    def forward(self, x0):
        x1 = self.modulelist1030(x0)
        return x1


class Bottleneck0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Bottleneck0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.bn0 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn1 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn2 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Bottleneck1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn0 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn1 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn2 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.backbone_layer1_0_downsample0 = Backbone_layer1_0_downsample(
            params=params)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.backbone_layer1_0_downsample0(x0)
        x10 = x8 + x9
        x11 = self.relu2(x10)
        return x11


class Backbone_transition1_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition1_1, self).__init__()
        self.backbone_transition1_1_00 = Backbone_transition1_1_0(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition1_1_00(x0)
        return x1


class Backbone_transition2_2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition2_2, self).__init__()
        self.backbone_transition2_2_00 = Backbone_transition2_2_0(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition2_2_00(x0)
        return x1


class Backbone_transition3_3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition3_3, self).__init__()
        self.backbone_transition3_3_00 = Backbone_transition3_3_0(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition3_3_00(x0)
        return x1


class Ocr_ocr_distri_head_conv_bn_dropout(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_ocr_distri_head_conv_bn_dropout, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=512,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=1024)
        self.ocr_ocr_distri_head_conv_bn_dropout_10 = Ocr_ocr_distri_head_conv_bn_dropout_1(
            params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_ocr_distri_head_conv_bn_dropout_10(x1)
        return x2


class Backbone_transition1_0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_transition1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=48,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.bn0 = layers.SyncBatchNorm(48, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        return x3


class Ocr_conv3x3_ocr_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_conv3x3_ocr_1, self).__init__()
        self.bn0 = layers.SyncBatchNorm(512, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class Ocr_aux_head_1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_aux_head_1, self).__init__()
        self.bn0 = layers.SyncBatchNorm(720, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()

    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.relu0(x1)
        return x2


class HighResolutionModule0(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule0, self).__init__()
        self.modulelist58_00 = ModuleList58_0(params=params)
        self.modulelist59_00 = ModuleList59_0(params=params)
        self.modulelist60_00 = ModuleList60_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist61_00 = ModuleList61_0(params=params)
        self.modulelist62_00 = ModuleList62_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist63_00 = ModuleList63_0(params=params)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2):
        x3, x4, x5 = self.modulelist58_00(x0, x1, x2)
        x6 = x3.shape[3]
        x7 = x3.shape[2]
        x8 = self.modulelist59_00(x4)
        x9 = [x7, x6]
        x10 = isinstance(x9, paddle.fluid.Variable)
        if x10:
            x9 = x9.numpy().tolist()
        assert None == None, 'The None must be None!'
        x13 = paddle.nn.functional.interpolate(
            x=x8,
            size=x9,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x14 = x3 + x13
        x15 = x3.shape[3]
        x16 = x3.shape[2]
        x17 = self.modulelist60_00(x5)
        x18 = [x16, x15]
        x19 = isinstance(x18, paddle.fluid.Variable)
        if x19:
            x18 = x18.numpy().tolist()
        assert None == None, 'The None must be None!'
        x22 = paddle.nn.functional.interpolate(
            x=x17,
            size=x18,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x23 = x14 + x22
        x24 = self.relu0(x23)
        x25 = self.modulelist61_00(x3)
        x26 = x25 + x4
        x27 = x4.shape[3]
        x28 = x4.shape[2]
        x29 = self.modulelist62_00(x5)
        x30 = [x28, x27]
        x31 = isinstance(x30, paddle.fluid.Variable)
        if x31:
            x30 = x30.numpy().tolist()
        assert None == None, 'The None must be None!'
        x34 = paddle.nn.functional.interpolate(
            x=x29,
            size=x30,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x35 = x26 + x34
        x36 = self.relu1(x35)
        x37, x38 = self.modulelist63_00(x3, x4)
        x39 = x37 + x38
        x40 = x39 + x5
        x41 = self.relu2(x40)
        x42 = (x24, x36, x41)
        x43, x44, x45 = x42
        return x43, x44, x45


class HighResolutionModule1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule1, self).__init__()
        self.modulelist70_00 = ModuleList70_0(params=params)
        self.modulelist71_00 = ModuleList71_0(params=params)
        self.modulelist72_00 = ModuleList72_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist73_00 = ModuleList73_0(params=params)
        self.modulelist74_00 = ModuleList74_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist75_00 = ModuleList75_0(params=params)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2):
        x3, x4, x5 = self.modulelist70_00(x0, x1, x2)
        x6 = x3.shape[3]
        x7 = x3.shape[2]
        x8 = self.modulelist71_00(x4)
        x9 = [x7, x6]
        x10 = isinstance(x9, paddle.fluid.Variable)
        if x10:
            x9 = x9.numpy().tolist()
        assert None == None, 'The None must be None!'
        x13 = paddle.nn.functional.interpolate(
            x=x8,
            size=x9,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x14 = x3 + x13
        x15 = x3.shape[3]
        x16 = x3.shape[2]
        x17 = self.modulelist72_00(x5)
        x18 = [x16, x15]
        x19 = isinstance(x18, paddle.fluid.Variable)
        if x19:
            x18 = x18.numpy().tolist()
        assert None == None, 'The None must be None!'
        x22 = paddle.nn.functional.interpolate(
            x=x17,
            size=x18,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x23 = x14 + x22
        x24 = self.relu0(x23)
        x25 = self.modulelist73_00(x3)
        x26 = x25 + x4
        x27 = x4.shape[3]
        x28 = x4.shape[2]
        x29 = self.modulelist74_00(x5)
        x30 = [x28, x27]
        x31 = isinstance(x30, paddle.fluid.Variable)
        if x31:
            x30 = x30.numpy().tolist()
        assert None == None, 'The None must be None!'
        x34 = paddle.nn.functional.interpolate(
            x=x29,
            size=x30,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x35 = x26 + x34
        x36 = self.relu1(x35)
        x37, x38 = self.modulelist75_00(x3, x4)
        x39 = x37 + x38
        x40 = x39 + x5
        x41 = self.relu2(x40)
        x42 = (x24, x36, x41)
        x43, x44, x45 = x42
        return x43, x44, x45


class HighResolutionModule2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule2, self).__init__()
        self.modulelist93_00 = ModuleList93_0(params=params)
        self.modulelist94_00 = ModuleList94_0(params=params)
        self.modulelist95_00 = ModuleList95_0(params=params)
        self.modulelist96_00 = ModuleList96_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist97_00 = ModuleList97_0(params=params)
        self.modulelist98_00 = ModuleList98_0(params=params)
        self.modulelist99_00 = ModuleList99_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist100_00 = ModuleList100_0(params=params)
        self.modulelist101_00 = ModuleList101_0(params=params)
        self.relu2 = paddle.nn.ReLU()
        self.modulelist102_00 = ModuleList102_0(params=params)
        self.modulelist103_00 = ModuleList103_0(params=params)
        self.relu3 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2, x3):
        x4, x5, x6, x7 = self.modulelist93_00(x0, x1, x2, x3)
        x8 = x4.shape[3]
        x9 = x4.shape[2]
        x10 = self.modulelist94_00(x5)
        x11 = [x9, x8]
        x12 = isinstance(x11, paddle.fluid.Variable)
        if x12:
            x11 = x11.numpy().tolist()
        assert None == None, 'The None must be None!'
        x15 = paddle.nn.functional.interpolate(
            x=x10,
            size=x11,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x16 = x4 + x15
        x17 = x4.shape[3]
        x18 = x4.shape[2]
        x19 = self.modulelist95_00(x6)
        x20 = [x18, x17]
        x21 = isinstance(x20, paddle.fluid.Variable)
        if x21:
            x20 = x20.numpy().tolist()
        assert None == None, 'The None must be None!'
        x24 = paddle.nn.functional.interpolate(
            x=x19,
            size=x20,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x25 = x16 + x24
        x26 = x4.shape[3]
        x27 = x4.shape[2]
        x28 = self.modulelist96_00(x7)
        x29 = [x27, x26]
        x30 = isinstance(x29, paddle.fluid.Variable)
        if x30:
            x29 = x29.numpy().tolist()
        assert None == None, 'The None must be None!'
        x33 = paddle.nn.functional.interpolate(
            x=x28,
            size=x29,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x34 = x25 + x33
        x35 = self.relu0(x34)
        x36 = self.modulelist97_00(x4)
        x37 = x36 + x5
        x38 = x5.shape[3]
        x39 = x5.shape[2]
        x40 = self.modulelist98_00(x6)
        x41 = [x39, x38]
        x42 = isinstance(x41, paddle.fluid.Variable)
        if x42:
            x41 = x41.numpy().tolist()
        assert None == None, 'The None must be None!'
        x45 = paddle.nn.functional.interpolate(
            x=x40,
            size=x41,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x46 = x37 + x45
        x47 = x5.shape[3]
        x48 = x5.shape[2]
        x49 = self.modulelist99_00(x7)
        x50 = [x48, x47]
        x51 = isinstance(x50, paddle.fluid.Variable)
        if x51:
            x50 = x50.numpy().tolist()
        assert None == None, 'The None must be None!'
        x54 = paddle.nn.functional.interpolate(
            x=x49,
            size=x50,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x55 = x46 + x54
        x56 = self.relu1(x55)
        x57, x58 = self.modulelist100_00(x4, x5)
        x59 = x57 + x58
        x60 = x59 + x6
        x61 = x6.shape[3]
        x62 = x6.shape[2]
        x63 = self.modulelist101_00(x7)
        x64 = [x62, x61]
        x65 = isinstance(x64, paddle.fluid.Variable)
        if x65:
            x64 = x64.numpy().tolist()
        assert None == None, 'The None must be None!'
        x68 = paddle.nn.functional.interpolate(
            x=x63,
            size=x64,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x69 = x60 + x68
        x70 = self.relu2(x69)
        x71, x72 = self.modulelist102_00(x4, x5)
        x73 = x71 + x72
        x74 = self.modulelist103_00(x6)
        x75 = x73 + x74
        x76 = x75 + x7
        x77 = self.relu3(x76)
        x78 = (x35, x56, x70, x77)
        x79, x80, x81, x82 = x78
        return x79, x80, x81, x82


class HighResolutionModule3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule3, self).__init__()
        self.modulelist55_00 = ModuleList55_0(params=params)
        self.modulelist56_00 = ModuleList56_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist57_00 = ModuleList57_0(params=params)
        self.relu1 = paddle.nn.ReLU()

    def forward(self, x0, x1):
        x2, x3 = self.modulelist55_00(x0, x1)
        x4 = x2.shape[3]
        x5 = x2.shape[2]
        x6 = self.modulelist56_00(x3)
        x7 = [x5, x4]
        x8 = isinstance(x7, paddle.fluid.Variable)
        if x8:
            x7 = x7.numpy().tolist()
        assert None == None, 'The None must be None!'
        x11 = paddle.nn.functional.interpolate(
            x=x6,
            size=x7,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x12 = x2 + x11
        x13 = self.relu0(x12)
        x14 = self.modulelist57_00(x2)
        x15 = x14 + x3
        x16 = self.relu1(x15)
        x17 = (x16, x13)
        x18, x19 = x17
        x20 = (x18, x19)
        x21, x22 = x20
        return x21, x22


class HighResolutionModule4(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule4, self).__init__()
        self.modulelist64_00 = ModuleList64_0(params=params)
        self.modulelist65_00 = ModuleList65_0(params=params)
        self.modulelist66_00 = ModuleList66_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist67_00 = ModuleList67_0(params=params)
        self.modulelist68_00 = ModuleList68_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist69_00 = ModuleList69_0(params=params)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2):
        x3, x4, x5 = self.modulelist64_00(x0, x1, x2)
        x6 = x3.shape[3]
        x7 = x3.shape[2]
        x8 = self.modulelist65_00(x4)
        x9 = [x7, x6]
        x10 = isinstance(x9, paddle.fluid.Variable)
        if x10:
            x9 = x9.numpy().tolist()
        assert None == None, 'The None must be None!'
        x13 = paddle.nn.functional.interpolate(
            x=x8,
            size=x9,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x14 = x3 + x13
        x15 = x3.shape[3]
        x16 = x3.shape[2]
        x17 = self.modulelist66_00(x5)
        x18 = [x16, x15]
        x19 = isinstance(x18, paddle.fluid.Variable)
        if x19:
            x18 = x18.numpy().tolist()
        assert None == None, 'The None must be None!'
        x22 = paddle.nn.functional.interpolate(
            x=x17,
            size=x18,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x23 = x14 + x22
        x24 = self.relu0(x23)
        x25 = self.modulelist67_00(x3)
        x26 = x25 + x4
        x27 = x4.shape[3]
        x28 = x4.shape[2]
        x29 = self.modulelist68_00(x5)
        x30 = [x28, x27]
        x31 = isinstance(x30, paddle.fluid.Variable)
        if x31:
            x30 = x30.numpy().tolist()
        assert None == None, 'The None must be None!'
        x34 = paddle.nn.functional.interpolate(
            x=x29,
            size=x30,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x35 = x26 + x34
        x36 = self.relu1(x35)
        x37, x38 = self.modulelist69_00(x3, x4)
        x39 = x37 + x38
        x40 = x39 + x5
        x41 = self.relu2(x40)
        x42 = (x24, x36, x41)
        x43, x44, x45 = x42
        return x43, x44, x45


class HighResolutionModule5(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule5, self).__init__()
        self.modulelist1040 = ModuleList104(params=params)
        self.modulelist1050 = ModuleList105(params=params)
        self.modulelist1060 = ModuleList106(params=params)
        self.modulelist1070 = ModuleList107(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist1080 = ModuleList108(params=params)
        self.modulelist1090 = ModuleList109(params=params)
        self.modulelist1100 = ModuleList110(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist1110 = ModuleList111(params=params)
        self.modulelist1120 = ModuleList112(params=params)
        self.relu2 = paddle.nn.ReLU()
        self.modulelist1130 = ModuleList113(params=params)
        self.modulelist1140 = ModuleList114(params=params)
        self.relu3 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2, x3):
        x4, x5, x6, x7 = self.modulelist1040(x0, x1, x2, x3)
        x8 = x4.shape[3]
        x9 = x4.shape[2]
        x10 = self.modulelist1050(x5)
        x11 = [x9, x8]
        x12 = isinstance(x11, paddle.fluid.Variable)
        if x12:
            x11 = x11.numpy().tolist()
        assert None == None, 'The None must be None!'
        x15 = paddle.nn.functional.interpolate(
            x=x10,
            size=x11,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x16 = x4 + x15
        x17 = x4.shape[3]
        x18 = x4.shape[2]
        x19 = self.modulelist1060(x6)
        x20 = [x18, x17]
        x21 = isinstance(x20, paddle.fluid.Variable)
        if x21:
            x20 = x20.numpy().tolist()
        assert None == None, 'The None must be None!'
        x24 = paddle.nn.functional.interpolate(
            x=x19,
            size=x20,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x25 = x16 + x24
        x26 = x4.shape[3]
        x27 = x4.shape[2]
        x28 = self.modulelist1070(x7)
        x29 = [x27, x26]
        x30 = isinstance(x29, paddle.fluid.Variable)
        if x30:
            x29 = x29.numpy().tolist()
        assert None == None, 'The None must be None!'
        x33 = paddle.nn.functional.interpolate(
            x=x28,
            size=x29,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x34 = x25 + x33
        x35 = self.relu0(x34)
        x36 = self.modulelist1080(x4)
        x37 = x36 + x5
        x38 = x5.shape[3]
        x39 = x5.shape[2]
        x40 = self.modulelist1090(x6)
        x41 = [x39, x38]
        x42 = isinstance(x41, paddle.fluid.Variable)
        if x42:
            x41 = x41.numpy().tolist()
        assert None == None, 'The None must be None!'
        x45 = paddle.nn.functional.interpolate(
            x=x40,
            size=x41,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x46 = x37 + x45
        x47 = x5.shape[3]
        x48 = x5.shape[2]
        x49 = self.modulelist1100(x7)
        x50 = [x48, x47]
        x51 = isinstance(x50, paddle.fluid.Variable)
        if x51:
            x50 = x50.numpy().tolist()
        assert None == None, 'The None must be None!'
        x54 = paddle.nn.functional.interpolate(
            x=x49,
            size=x50,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x55 = x46 + x54
        x56 = self.relu1(x55)
        x57, x58 = self.modulelist1110(x4, x5)
        x59 = x57 + x58
        x60 = x59 + x6
        x61 = x6.shape[3]
        x62 = x6.shape[2]
        x63 = self.modulelist1120(x7)
        x64 = [x62, x61]
        x65 = isinstance(x64, paddle.fluid.Variable)
        if x65:
            x64 = x64.numpy().tolist()
        assert None == None, 'The None must be None!'
        x68 = paddle.nn.functional.interpolate(
            x=x63,
            size=x64,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x69 = x60 + x68
        x70 = self.relu2(x69)
        x71, x72 = self.modulelist1130(x4, x5)
        x73 = x71 + x72
        x74 = self.modulelist1140(x6)
        x75 = x73 + x74
        x76 = x75 + x7
        x77 = self.relu3(x76)
        x78 = (x35, x56, x70, x77)
        x79, x80, x81, x82 = x78
        x83 = (x79, x80, x81, x82)
        x84, x85, x86, x87 = x83
        return x84, x85, x86, x87


class HighResolutionModule6(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule6, self).__init__()
        self.modulelist82_00 = ModuleList82_0(params=params)
        self.modulelist83_00 = ModuleList83_0(params=params)
        self.modulelist84_00 = ModuleList84_0(params=params)
        self.modulelist85_00 = ModuleList85_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist86_00 = ModuleList86_0(params=params)
        self.modulelist87_00 = ModuleList87_0(params=params)
        self.modulelist88_00 = ModuleList88_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist89_00 = ModuleList89_0(params=params)
        self.modulelist90_00 = ModuleList90_0(params=params)
        self.relu2 = paddle.nn.ReLU()
        self.modulelist91_00 = ModuleList91_0(params=params)
        self.modulelist92_00 = ModuleList92_0(params=params)
        self.relu3 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2, x3):
        x4, x5, x6, x7 = self.modulelist82_00(x0, x1, x2, x3)
        x8 = x4.shape[3]
        x9 = x4.shape[2]
        x10 = self.modulelist83_00(x5)
        x11 = [x9, x8]
        x12 = isinstance(x11, paddle.fluid.Variable)
        if x12:
            x11 = x11.numpy().tolist()
        assert None == None, 'The None must be None!'
        x15 = paddle.nn.functional.interpolate(
            x=x10,
            size=x11,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x16 = x4 + x15
        x17 = x4.shape[3]
        x18 = x4.shape[2]
        x19 = self.modulelist84_00(x6)
        x20 = [x18, x17]
        x21 = isinstance(x20, paddle.fluid.Variable)
        if x21:
            x20 = x20.numpy().tolist()
        assert None == None, 'The None must be None!'
        x24 = paddle.nn.functional.interpolate(
            x=x19,
            size=x20,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x25 = x16 + x24
        x26 = x4.shape[3]
        x27 = x4.shape[2]
        x28 = self.modulelist85_00(x7)
        x29 = [x27, x26]
        x30 = isinstance(x29, paddle.fluid.Variable)
        if x30:
            x29 = x29.numpy().tolist()
        assert None == None, 'The None must be None!'
        x33 = paddle.nn.functional.interpolate(
            x=x28,
            size=x29,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x34 = x25 + x33
        x35 = self.relu0(x34)
        x36 = self.modulelist86_00(x4)
        x37 = x36 + x5
        x38 = x5.shape[3]
        x39 = x5.shape[2]
        x40 = self.modulelist87_00(x6)
        x41 = [x39, x38]
        x42 = isinstance(x41, paddle.fluid.Variable)
        if x42:
            x41 = x41.numpy().tolist()
        assert None == None, 'The None must be None!'
        x45 = paddle.nn.functional.interpolate(
            x=x40,
            size=x41,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x46 = x37 + x45
        x47 = x5.shape[3]
        x48 = x5.shape[2]
        x49 = self.modulelist88_00(x7)
        x50 = [x48, x47]
        x51 = isinstance(x50, paddle.fluid.Variable)
        if x51:
            x50 = x50.numpy().tolist()
        assert None == None, 'The None must be None!'
        x54 = paddle.nn.functional.interpolate(
            x=x49,
            size=x50,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x55 = x46 + x54
        x56 = self.relu1(x55)
        x57, x58 = self.modulelist89_00(x4, x5)
        x59 = x57 + x58
        x60 = x59 + x6
        x61 = x6.shape[3]
        x62 = x6.shape[2]
        x63 = self.modulelist90_00(x7)
        x64 = [x62, x61]
        x65 = isinstance(x64, paddle.fluid.Variable)
        if x65:
            x64 = x64.numpy().tolist()
        assert None == None, 'The None must be None!'
        x68 = paddle.nn.functional.interpolate(
            x=x63,
            size=x64,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x69 = x60 + x68
        x70 = self.relu2(x69)
        x71, x72 = self.modulelist91_00(x4, x5)
        x73 = x71 + x72
        x74 = self.modulelist92_00(x6)
        x75 = x73 + x74
        x76 = x75 + x7
        x77 = self.relu3(x76)
        x78 = (x35, x56, x70, x77)
        x79, x80, x81, x82 = x78
        return x79, x80, x81, x82


class HighResolutionModule7(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionModule7, self).__init__()
        self.modulelist76_00 = ModuleList76_0(params=params)
        self.modulelist77_00 = ModuleList77_0(params=params)
        self.modulelist78_00 = ModuleList78_0(params=params)
        self.relu0 = paddle.nn.ReLU()
        self.modulelist79_00 = ModuleList79_0(params=params)
        self.modulelist80_00 = ModuleList80_0(params=params)
        self.relu1 = paddle.nn.ReLU()
        self.modulelist81_00 = ModuleList81_0(params=params)
        self.relu2 = paddle.nn.ReLU()

    def forward(self, x0, x1, x2):
        x3, x4, x5 = self.modulelist76_00(x0, x1, x2)
        x6 = x3.shape[3]
        x7 = x3.shape[2]
        x8 = self.modulelist77_00(x4)
        x9 = [x7, x6]
        x10 = isinstance(x9, paddle.fluid.Variable)
        if x10:
            x9 = x9.numpy().tolist()
        assert None == None, 'The None must be None!'
        x13 = paddle.nn.functional.interpolate(
            x=x8,
            size=x9,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x14 = x3 + x13
        x15 = x3.shape[3]
        x16 = x3.shape[2]
        x17 = self.modulelist78_00(x5)
        x18 = [x16, x15]
        x19 = isinstance(x18, paddle.fluid.Variable)
        if x19:
            x18 = x18.numpy().tolist()
        assert None == None, 'The None must be None!'
        x22 = paddle.nn.functional.interpolate(
            x=x17,
            size=x18,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x23 = x14 + x22
        x24 = self.relu0(x23)
        x25 = self.modulelist79_00(x3)
        x26 = x25 + x4
        x27 = x4.shape[3]
        x28 = x4.shape[2]
        x29 = self.modulelist80_00(x5)
        x30 = [x28, x27]
        x31 = isinstance(x30, paddle.fluid.Variable)
        if x31:
            x30 = x30.numpy().tolist()
        assert None == None, 'The None must be None!'
        x34 = paddle.nn.functional.interpolate(
            x=x29,
            size=x30,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x35 = x26 + x34
        x36 = self.relu1(x35)
        x37, x38 = self.modulelist81_00(x3, x4)
        x39 = x37 + x38
        x40 = x39 + x5
        x41 = self.relu2(x40)
        x42 = (x41, x24, x36)
        x43, x44, x45 = x42
        x46 = (x43, x44, x45)
        x47, x48, x49 = x46
        return x47, x48, x49


class ObjectAttentionBlock(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ObjectAttentionBlock, self).__init__()
        self.ocr_ocr_distri_head_object_context_block_f_pixel0 = Ocr_ocr_distri_head_object_context_block_f_pixel(
            params=params)
        self.ocr_ocr_distri_head_object_context_block_f_object0 = Ocr_ocr_distri_head_object_context_block_f_object(
            params=params)
        self.ocr_ocr_distri_head_object_context_block_f_down0 = Ocr_ocr_distri_head_object_context_block_f_down(
            params=params)
        self.softmax0 = paddle.nn.Softmax(axis=-1)
        self.ocr_ocr_distri_head_object_context_block_f_up0 = Ocr_ocr_distri_head_object_context_block_f_up(
            params=params)

    def forward(self, x0, x2, x10):
        x1 = self.ocr_ocr_distri_head_object_context_block_f_pixel0(x0)
        x3 = [x2, 256, -1]
        x4 = x1.dtype
        x4 = str(x4)
        x5 = x4 == 'VarType.BOOL'
        if x5:
            x1 = fluid.layers.cast(x=x1, dtype='int32')
        x7 = fluid.layers.reshape(x=x1, shape=x3)
        if x5:
            x7 = fluid.layers.cast(x=x7, dtype='bool')
        x9 = fluid.layers.transpose(x=x7, perm=[0, 2, 1])
        x11 = self.ocr_ocr_distri_head_object_context_block_f_object0(x10)
        x12 = [x2, 256, -1]
        x13 = x11.dtype
        x13 = str(x13)
        x14 = x13 == 'VarType.BOOL'
        if x14:
            x11 = fluid.layers.cast(x=x11, dtype='int32')
        x16 = fluid.layers.reshape(x=x11, shape=x12)
        if x14:
            x16 = fluid.layers.cast(x=x16, dtype='bool')
        x18 = self.ocr_ocr_distri_head_object_context_block_f_down0(x10)
        x19 = [x2, 256, -1]
        x20 = x18.dtype
        x20 = str(x20)
        x21 = x20 == 'VarType.BOOL'
        if x21:
            x18 = fluid.layers.cast(x=x18, dtype='int32')
        x23 = fluid.layers.reshape(x=x18, shape=x19)
        if x21:
            x23 = fluid.layers.cast(x=x23, dtype='bool')
        x25 = fluid.layers.transpose(x=x23, perm=[0, 2, 1])
        x26 = paddle.matmul(x=x9, y=x16)
        x27 = x26 * 0.0625
        x28 = self.softmax0(x27)
        x29 = paddle.matmul(x=x28, y=x25)
        x30 = fluid.layers.transpose(x=x29, perm=[0, 2, 1])
        x31 = x30
        x32 = x0.shape[2]
        x33 = x0.shape[3]
        x34 = [x2, 256, x32, x33]
        x35 = x31.dtype
        x35 = str(x35)
        x36 = x35 == 'VarType.BOOL'
        if x36:
            x31 = fluid.layers.cast(x=x31, dtype='int32')
        x38 = fluid.layers.reshape(x=x31, shape=x34)
        if x36:
            x38 = fluid.layers.cast(x=x38, dtype='bool')
        x40 = self.ocr_ocr_distri_head_object_context_block_f_up0(x38)
        return x40


class Backbone_stage3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage3, self).__init__()
        self.highresolutionmodule00 = HighResolutionModule0(params=params)
        self.highresolutionmodule40 = HighResolutionModule4(params=params)
        self.highresolutionmodule10 = HighResolutionModule1(params=params)
        self.highresolutionmodule70 = HighResolutionModule7(params=params)

    def forward(self, x0, x1, x2):
        x3, x4, x5 = self.highresolutionmodule00(x0, x1, x2)
        x6, x7, x8 = self.highresolutionmodule40(x3, x4, x5)
        x9, x10, x11 = self.highresolutionmodule10(x6, x7, x8)
        x12, x13, x14 = self.highresolutionmodule70(x9, x10, x11)
        return x12, x13, x14


class Backbone_layer1(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_layer1, self).__init__()
        self.bottleneck10 = Bottleneck1(params=params)
        self.bottleneck00 = Bottleneck0(params=params)
        self.bottleneck01 = Bottleneck0(params=params)
        self.bottleneck02 = Bottleneck0(params=params)

    def forward(self, x0):
        x1 = self.bottleneck10(x0)
        x2 = self.bottleneck00(x1)
        x3 = self.bottleneck01(x2)
        x4 = self.bottleneck02(x3)
        return x4


class Ocr_aux_head(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_aux_head, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=720,
            kernel_size=(1, 1),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=720)
        self.ocr_aux_head_10 = Ocr_aux_head_1(params=params)
        self.conv1 = paddle.nn.Conv2D(
            out_channels=19,
            kernel_size=(1, 1),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=720)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_aux_head_10(x1)
        x3 = self.conv1(x2)
        return x3


class Backbone_stage4(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage4, self).__init__()
        self.highresolutionmodule60 = HighResolutionModule6(params=params)
        self.highresolutionmodule20 = HighResolutionModule2(params=params)
        self.highresolutionmodule50 = HighResolutionModule5(params=params)

    def forward(self, x0, x1, x2, x3):
        x4, x5, x6, x7 = self.highresolutionmodule60(x0, x1, x2, x3)
        x8, x9, x10, x11 = self.highresolutionmodule20(x4, x5, x6, x7)
        x12, x13, x14, x15 = self.highresolutionmodule50(x8, x9, x10, x11)
        return x12, x13, x14, x15


class Ocr_conv3x3_ocr(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Ocr_conv3x3_ocr, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=512,
            kernel_size=(3, 3),
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=720)
        self.ocr_conv3x3_ocr_10 = Ocr_conv3x3_ocr_1(params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.ocr_conv3x3_ocr_10(x1)
        return x2


class Backbone_stage2(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Backbone_stage2, self).__init__()
        self.highresolutionmodule30 = HighResolutionModule3(params=params)

    def forward(self, x0, x1):
        x2, x3 = self.highresolutionmodule30(x0, x1)
        return x2, x3


class ModuleList3(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList3, self).__init__()
        self.backbone_transition1_00 = Backbone_transition1_0(params=params)
        self.backbone_transition1_10 = Backbone_transition1_1(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition1_00(x0)
        x2 = self.backbone_transition1_10(x0)
        return x1, x2


class ModuleList4(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList4, self).__init__()
        self.backbone_transition2_20 = Backbone_transition2_2(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition2_20(x0)
        return x1


class ModuleList5(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(ModuleList5, self).__init__()
        self.backbone_transition3_30 = Backbone_transition3_3(params=params)

    def forward(self, x0):
        x1 = self.backbone_transition3_30(x0)
        return x1


class SpatialOCR_Module(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(SpatialOCR_Module, self).__init__()
        self.objectattentionblock0 = ObjectAttentionBlock(params=params)
        self.ocr_ocr_distri_head_conv_bn_dropout0 = Ocr_ocr_distri_head_conv_bn_dropout(
            params=params)

    def forward(self, x0, x2):
        x1 = x0.shape[0]
        x3 = self.objectattentionblock0(x0, x1, x2)
        x4 = [x3, x0]
        x5 = fluid.layers.concat(input=x4, axis=1)
        x6 = self.ocr_ocr_distri_head_conv_bn_dropout0(x5)
        return x6


class HighResolutionNet(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(HighResolutionNet, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=3)
        self.bn0 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            out_channels=64,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=64)
        self.bn1 = layers.SyncBatchNorm(64, momentum=0.9, epsilon=1e-05)
        self.relu1 = paddle.nn.ReLU()
        self.backbone_layer10 = Backbone_layer1(params=params)
        self.modulelist30 = ModuleList3(params=params)
        self.backbone_stage20 = Backbone_stage2(params=params)
        self.modulelist40 = ModuleList4(params=params)
        self.backbone_stage30 = Backbone_stage3(params=params)
        self.modulelist50 = ModuleList5(params=params)
        self.backbone_stage40 = Backbone_stage4(params=params)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.backbone_layer10(x6)
        x8, x9 = self.modulelist30(x7)
        x10, x11 = self.backbone_stage20(x8, x9)
        x12 = self.modulelist40(x10)
        x13, x14, x15 = self.backbone_stage30(x11, x10, x12)
        x16 = self.modulelist50(x13)
        x17, x18, x19, x20 = self.backbone_stage40(x14, x15, x13, x16)
        x21 = x17.shape[2]
        x22 = x17.shape[3]
        x23 = [x21, x22]
        x24 = isinstance(x23, paddle.fluid.Variable)
        if x24:
            x23 = x23.numpy().tolist()
        assert None == None, 'The None must be None!'
        x27 = paddle.nn.functional.interpolate(
            x=x18,
            size=x23,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x28 = [x21, x22]
        x29 = isinstance(x28, paddle.fluid.Variable)
        if x29:
            x28 = x28.numpy().tolist()
        assert None == None, 'The None must be None!'
        x32 = paddle.nn.functional.interpolate(
            x=x19,
            size=x28,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x33 = [x21, x22]
        x34 = isinstance(x33, paddle.fluid.Variable)
        if x34:
            x33 = x33.numpy().tolist()
        assert None == None, 'The None must be None!'
        x37 = paddle.nn.functional.interpolate(
            x=x20,
            size=x33,
            align_corners=False,
            align_mode=0,
            mode='bilinear',
            scale_factor=None)
        x38 = [x17, x27, x32, x37]
        x39 = fluid.layers.concat(input=x38, axis=1)
        return x39


class OCR_block(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(OCR_block, self).__init__()
        self.ocr_conv3x3_ocr0 = Ocr_conv3x3_ocr(params=params)
        self.ocr_aux_head0 = Ocr_aux_head(params=params)
        self.softmax0 = paddle.nn.Softmax(axis=2)
        self.spatialocr_module0 = SpatialOCR_Module(params=params)
        self.conv0 = paddle.nn.Conv2D(
            out_channels=19,
            kernel_size=(1, 1),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=512)

    def forward(self, x0):
        x1 = self.ocr_conv3x3_ocr0(x0)
        x2 = self.ocr_aux_head0(x0)
        x3 = x2.shape[0]
        x4 = x2.shape[1]
        x5 = [x3, x4, -1]
        x6 = x2.dtype
        x6 = str(x6)
        x7 = x6 == 'VarType.BOOL'
        if x7:
            x2 = fluid.layers.cast(x=x2, dtype='int32')
        x9 = fluid.layers.reshape(x=x2, shape=x5)
        if x7:
            x9 = fluid.layers.cast(x=x9, dtype='bool')
        x11 = x1.shape[1]
        x12 = [x3, x11, -1]
        x13 = x1.dtype
        x13 = str(x13)
        x14 = x13 == 'VarType.BOOL'
        if x14:
            x1 = fluid.layers.cast(x=x1, dtype='int32')
        x16 = fluid.layers.reshape(x=x1, shape=x12)
        if x14:
            x16 = fluid.layers.cast(x=x16, dtype='bool')
        x18 = fluid.layers.transpose(x=x16, perm=[0, 2, 1])
        x19 = x9 * 1
        x20 = self.softmax0(x19)
        x21 = paddle.matmul(x=x20, y=x18)
        x22 = fluid.layers.transpose(x=x21, perm=[0, 2, 1])
        x23 = paddle.tensor.unsqueeze(x=x22, axis=3)
        x24 = self.spatialocr_module0(x1, x23)
        x25 = self.conv0(x24)
        x26 = (x24, x2, x25)
        x27, x28, x29 = x26
        return x27, x28, x29  # x27 -> ocr_feats  x29 -> cls_out


class Scale_attn(fluid.dygraph.Layer):
    def __init__(
            self,
            params,
    ):
        super(Scale_attn, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=512)
        self.bn0 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            out_channels=256,
            kernel_size=(3, 3),
            bias_attr=False,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            in_channels=256)
        self.bn1 = layers.SyncBatchNorm(256, momentum=0.9, epsilon=1e-05)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            out_channels=1,
            kernel_size=(1, 1),
            bias_attr=False,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            in_channels=256)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = fluid.layers.sigmoid(x=x7)
        return x8
