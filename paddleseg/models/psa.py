"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import paddle
from paddle import nn
from collections import OrderedDict
from paddleseg.cvlibs.param_init import kaiming_uniform, kaiming_normal_init, constant_init
import paddle.nn.functional as F


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2D(
            in_dim,
            reduction_dim,
            kernel_size=3,
            dilation=dil,
            padding=dil,
            bias_attr=False,
            groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU())


class DPC(nn.Layer):
    '''
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    '''

    def __init__(self,
                 in_dim,
                 reduction_dim=256,
                 output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False,
                 separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = paddle.concat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


class AtrousSpatialPyramidPoolingModule(nn.Layer):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self,
                 in_dim,
                 reduction_dim=256,
                 output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2D(
                    in_dim, reduction_dim, kernel_size=1, bias_attr=False),
                Norm2d(reduction_dim),
                nn.ReLU()))
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias_attr=False),
                    Norm2d(reduction_dim),
                    nn.ReLU()))
        self.features = nn.LayerList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2D(
                in_dim, reduction_dim, kernel_size=1, bias_attr=False),
            Norm2d(reduction_dim),
            nn.ReLU())

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = paddle.concat((out, y), 1)
        return out


def BNReLU(ch):
    return nn.Sequential(Norm2d(ch), nn.ReLU())


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(
            high_level_ch, bottleneck_ch, output_stride=output_stride)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(
        x, size=size, mode='bilinear', align_corners=False)


def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    #layer = getattr(cfg.MODEL, 'BNFUNC')
    layer = paddle.nn.BatchNorm2D
    normalization_layer = layer(in_channels, **kwargs)
    return normalization_layer


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2D, nn.Linear)):
                kaiming_normal_init(module.weight)
                if module.bias is not None:
                    constant_init(module.bias, 0)
            elif isinstance(module, paddle.nn.BatchNorm2D):
                constant_init(module.weight, 1)
                constant_init(module.bias, 0)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    #y_size = y.size(2), y.size(3)
    y_size = y.shape[2], y.shape[3]
    '''
    注释掉
    if cfg.OPTIONS.TORCH_VERSION >= 1.5:
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=align_corners)
    else:
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=align_corners)
    '''
    x_scaled = paddle.nn.functional.interpolate(
        x, size=y_size, mode='bilinear', align_corners=False)
    return x_scaled


def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    
    if cfg.OPTIONS.TORCH_VERSION >= 1.5:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners, recompute_scale_factor=True)
    else:
        x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners)
    '''
    x_scaled = paddle.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return x_scaled


def make_attn_head(in_ch, out_ch):
    #bot_ch = cfg.MODEL.SEGATTN_BOT_CH
    bot_ch = 256
    #if cfg.MODEL.MSCALE_OLDARCH:
    if False:
        return old_make_attn_head(in_ch, bot_ch, out_ch)

    od = OrderedDict([('conv0', nn.Conv2D(
        in_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False)),
                      ('bn0', Norm2d(bot_ch)), ('re0', nn.ReLU())])

    #if cfg.MODEL.MSCALE_INNER_3x3:
    if True:
        od['conv1'] = nn.Conv2D(
            bot_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False)
        od['bn1'] = Norm2d(bot_ch)
        od['re1'] = nn.ReLU()

    #if cfg.MODEL.MSCALE_DROPOUT:
    if False:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2D(bot_ch, out_ch, kernel_size=1, bias_attr=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential()
    for key in od:
        attn_head.add_sublayer(key, od[key])

    # init_attn(attn_head)
    return attn_head


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'


class SpatialGather_Module(nn.Layer):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.

        Output:
          The correlation of every class map with every feature map
          shape = [n, num_feats, num_classes, 1]


    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        #batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), \
        #probs.size(3)
        batch_size, c, _, _ = probs.shape[0], probs.shape[1], probs.shape[2], \
            probs.shape[3]

        # each class image now a vector
        #probs = probs.view(batch_size, c, -1)
        probs = probs.reshape((batch_size, c, -1))

        #feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.reshape((batch_size, feats.shape[1], -1))

        #feats = feats.permute(0, 2, 1)  # batch x hw x c
        feats = feats.transpose((0, 2, 1))  # batch x hw x c

        #probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        probs = F.softmax(self.scale * probs, axis=2)  # batch x k x hw
        ocr_context = paddle.matmul(probs, feats)

        #ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        ocr_context = ocr_context.transpose((0, 2, 1)).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Layer):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    '''

    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2D(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.key_channels),
            nn.Conv2D(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.key_channels), )
        self.f_object = nn.Sequential(
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.key_channels),
            nn.Conv2D(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.key_channels), )
        self.f_down = nn.Sequential(
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.key_channels), )
        self.f_up = nn.Sequential(
            nn.Conv2D(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            BNReLU(self.in_channels), )

    def forward(self, x, proxy):
        #batch_size, h, w = x.size(0), x.size(2), x.size(3)
        batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]
        if self.scale > 1:
            x = self.pool(x)

        #query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = self.f_pixel(x).reshape((batch_size, self.key_channels, -1))
        #query = query.permute(0, 2, 1)
        query = query.transpose((0, 2, 1))

        #key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        key = self.f_object(proxy).reshape((batch_size, self.key_channels, -1))

        #value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).reshape((batch_size, self.key_channels, -1))

        #value = value.permute(0, 2, 1)
        value = value.transpose((0, 2, 1))
        sim_map = paddle.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        #sim_map = F.softmax(sim_map, dim=-1)
        sim_map = F.softmax(sim_map, axis=-1)
        # add bg context ...
        context = paddle.matmul(sim_map, value)
        #context = context.permute(0, 2, 1).contiguous()
        context = context.transpose((0, 2, 1))

        #context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = context.reshape((batch_size, self.key_channels, *x.shape[2:]))
        context = self.f_up(context)
        if self.scale > 1:
            #context = F.interpolate(input=context, size=(h, w), mode='bilinear',
            #align_corners=cfg.MODEL.ALIGN_CORNERS)
            context = F.interpolate(
                input=context,
                size=(h, w),
                mode='bilinear',
                align_corners=False)

        return context


class SpatialOCR_Module(nn.Layer):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels, scale)
        #if cfg.MODEL.OCR_ASPP:
        if False:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=256, output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2D(
                _in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias_attr=False),
            BNReLU(out_channels),
            nn.Dropout2D(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        #if cfg.MODEL.OCR_ASPP:
        if False:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(
                paddle.concat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(paddle.concat([context, feats], 1))

        return output


class OCR_block(nn.Layer):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, high_level_ch):
        super(OCR_block, self).__init__()

        #ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        #ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        ocr_mid_channels = 512
        ocr_key_channels = 256
        #改了numclasses
        num_classes = 19

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2D(
                high_level_ch,
                ocr_mid_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            BNReLU(ocr_mid_channels), )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05, )
        self.cls_head = nn.Conv2D(
            ocr_mid_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)

        self.aux_head = nn.Sequential(
            nn.Conv2D(
                high_level_ch,
                high_level_ch,
                kernel_size=1,
                stride=1,
                padding=0),
            BNReLU(high_level_ch),
            nn.Conv2D(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True))

        #if cfg.OPTIONS.INIT_DECODER:
        if False:
            initialize_weights(self.conv3x3_ocr, self.ocr_gather_head,
                               self.ocr_distri_head, self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


from paddleseg.cvlibs import manager


@manager.MODELS.add_component
class MscaleOCR(nn.Layer):
    """
    OCR net
    """

    def __init__(self, num_classes, backbone, trunk='hrnetv2', criterion=None):
        super(MscaleOCR, self).__init__()
        #self.criterion = criterion
        #self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.backbone = backbone
        high_level_ch = self.backbone.high_level_ch
        #self.backbone = backone
        self.ocr = OCR_block(high_level_ch)
        self.scale_attn = make_attn_head(in_ch=512, out_ch=1)

    def _fwd(self, x):
        #x_size = x.size()[2:]
        x_size = x.shape[2:]
        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        return {'cls_out': cls_out, 'aux_out': aux_out, 'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
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

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        #x_1x = inputs['images']
        x_1x = inputs

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        if self.training:
            #assert 'gts' in inputs
            #gts = inputs['gts']
            #gts = inputs[1]
            #loss = 0.4 * self.criterion(aux, gts) + \
            #self.criterion(pred, gts)
            #return loss
            return [aux, pred]
        else:
            #output_dict['pred'] = pred
            #return output_dict
            return [pred]

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """

        #assert 'images' in inputs
        #x_1x = inputs['images']
        x_1x = inputs
        x_lo = ResizeX(x_1x, 0.5)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
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

        if self.training:
            #gts = inputs['gts']
            #gts = inputs[1]
            #do_rmi = False
            #aux_loss = self.criterion(joint_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix

            #do_rmi_main = True  # cfg.EPOCH > 0

            ######################################do_rmi=True###########
            #main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)

            #loss = 0.4 * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            #if cfg.LOSS.SUPERVISED_MSCALE_WT:
            if True:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                #loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                #loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                #loss += 0.05 * loss_lo
                #loss += 0.05 * loss_hi
            #return loss
            return [joint_aux, joint_pred, scaled_pred_05x, pred_10x]
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
            }
            #return output_dict
            return [joint_pred]

    def forward(self, inputs):

        if [0.5, 1.0, 2.0] and not self.training:
            return self.nscale_forward(inputs, [0.5, 1.0, 2.0])

        return self.two_scale_forward(inputs)


def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)
