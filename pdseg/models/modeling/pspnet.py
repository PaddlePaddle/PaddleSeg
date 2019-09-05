# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import avg_pool , conv, bn
from models.backbone.resnet import ResNet as resnet_backbone
from utils.config import cfg

def get_logit_interp(input, num_classes, out_shape, name="logit"):
    param_attr = fluid.ParamAttr(
        name=name + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))

    with scope(name):
        logit = conv(
            input,
            num_classes,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=True,
            name=name+'.conv2d.output.1')
        logit_interp = fluid.layers.resize_bilinear(
                    logit, 
                    out_shape=out_shape,
                    name='logit_interp') 
    return logit_interp


def psp_module(input, out_features):
    cat_layers = []
    sizes = (1,2,3,6)
    for size in sizes:
        psp_name = "psp_conv" + str(size)
        with scope(psp_name):
            pool = fluid.layers.adaptive_pool2d(input, 
                pool_size=[size, size], 
                pool_type='avg', 
                name=psp_name+'_adapool')
            data = conv(pool, out_features, filter_size=1, bias_attr=True, 
                    name= psp_name + '.conv2d.output.1')
            data_bn = bn(data, act='relu')
            interp = fluid.layers.resize_bilinear(data_bn, 
                out_shape=input.shape[2:], 
                name=psp_name+'_interp') 
        cat_layers.append(interp)
    cat_layers = [input] + cat_layers[::-1]
    cat = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')
    with scope("psp_conv_end"):
        data = conv(cat, 
                out_features, 
                filter_size=3,
                padding=1, 
                bias_attr=True,
                name='psp_conv_end.conv2d.output.1')
        out = bn(data, act='relu')

    return out

def resnet(input):
    # PSPNET backbone: resnet, ĬÈresnet50
    # end_points: resnetÖֹ²ã

    scale = cfg.MODEL.PSPNET.DEPTH_MULTIPLIER
    layers = cfg.MODEL.PSPNET.LAYERS
    end_points = layers - 1
    dilation_dict = {2:2, 3:4}
    model = resnet_backbone(layers, scale, stem='pspnet')
    data, _ = model.net(input, end_points=end_points, dilation_dict=dilation_dict)

    return data

def pspnet(input, num_classes):
    res = resnet(input)
    psp = psp_module(res, 512)
    #dropout = fluid.layers.dropout(psp, dropout_prob=0.1, name="dropout")
    logit = get_logit_interp(psp, num_classes, input.shape[2:])
    return logit


