"""
Custom Norm wrappers to enable sync BN, regular BN and for weight
initialization
"""
import re
import paddle
import paddle.nn as nn
#from config import cfg
from .para_init import kaiming_uniform_,kaiming_normal_,constant_



#from runx.logx import logx


#align_corners = cfg.MODEL.ALIGN_CORNERS
align_corners = False


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
                kaiming_normal_(module.weight)
                if module.bias is not None:
                    constant_(module.bias,0)
            elif isinstance(module, paddle.nn.BatchNorm2D):
                constant_(module.weight,1)
                constant_(module.bias,0)


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=align_corners)



def Upsample2(x):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                     align_corners=align_corners)


def Down2x(x):
    return paddle.nn.functional.interpolate(
        x, scale_factor=0.5, mode='bilinear', align_corners=align_corners)


def Up15x(x):
    return paddle.nn.functional.interpolate(
        x, scale_factor=1.5, mode='bilinear', align_corners=align_corners)


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
            x, size=y_size, mode='bilinear',
            align_corners=align_corners)
    return x_scaled


def DownX(x, scale_factor):
    '''
    scale x to the same size as y
    '''
    '''
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
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=align_corners)
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
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=align_corners)
    return x_scaled
