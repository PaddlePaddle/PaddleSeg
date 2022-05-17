# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# 
# This code is from: https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

#from config import cfg
from .utils import BNReLU
from .utils import get_aspp

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
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.key_channels),
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.key_channels),
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2D(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2D(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias_attr=False),
            BNReLU(self.in_channels),
        )

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
            context = F.interpolate(input=context, size=(h, w), mode='bilinear',
                                    align_corners=False)

        return context


class SpatialOCR_Module(nn.Layer):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1,
                 dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels,
                                                         scale)
        #if cfg.MODEL.OCR_ASPP:
        if False:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=256,
                output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2D(_in_channels, out_channels, kernel_size=1, padding=0,
                      bias_attr=False),
            BNReLU(out_channels),
            nn.Dropout2D(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        #if cfg.MODEL.OCR_ASPP:
        if False:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(paddle.concat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(paddle.concat([context, feats], 1))

        return output
