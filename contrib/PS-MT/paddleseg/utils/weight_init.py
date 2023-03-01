import os
import PIL
import numpy as np
import paddle
import paddle.nn as nn
from paddleseg.cvlibs import param_init
from PIL import Image


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for m in feature.sublayers():
        if isinstance(m, (nn.Conv1D, nn.Conv2D, nn.Conv3D)):
            conv_init(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            param_init.constant_init(m.weight, value=1)
            param_init.constant_init(m.bias, value=0)
    if len(feature.sublayers()) == 0:
        m = feature
        if isinstance(m, (nn.Conv1D, nn.Conv2D, nn.Conv3D)):
            conv_init(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            param_init.constant_init(m.weight, value=1)
            param_init.constant_init(m.bias, value=0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr, clear_flag=False):
    group_decay = []
    group_no_decay = []

    for m in module.sublayers():
        if len(module.sublayers()) == 0:
            m = module
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1D, nn.Conv2D, nn.Conv3D, nn.Conv2DTranspose, nn.Conv3DTranspose)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        # elif isinstance(m, Conv2_5D_depth):
        #     group_decay.append(m.weight_0)
        #     group_decay.append(m.weight_1)
        #     group_decay.append(m.weight_2)
        #     if m.bias is not None:
        #         group_no_decay.append(m.bias)
        # elif isinstance(m, Conv2_5D_disp):
        #     group_decay.append(m.weight_0)
        #     group_decay.append(m.weight_1)
        #     group_decay.append(m.weight_2)
        #     if m.bias is not None:
        #         group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1D) or isinstance(m, nn.BatchNorm2D) \
                or isinstance(m, nn.BatchNorm3D) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.ParameterList):
            group_decay.append(m)
        elif isinstance(m, nn.Embedding):
            group_decay.append(m)
    if clear_flag:
        weight_group.append(dict(params=group_decay))
        weight_group.append(dict(params=group_no_decay))
    else:
        weight_group.append(dict(params=group_decay, learning_rate=lr))
        weight_group.append(dict(params=group_no_decay, weight_decay=.0, learning_rate=lr))

    return weight_group


def get_voc_pallete(num_classes):
    n = num_classes
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            paddle.add(paddle.multiply(t, paddle.to_tensor(s)), paddle.to_tensor(m))
        return tensor


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_weights(*models):
    for model in models:
        for m in model.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight, fan_in='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1)
                param_init.constant_init(m.bias, value=0)
            elif isinstance(m, nn.Linear):
                param_init.normal_init(m.weight, mean=0.0, std=0.01)
                param_init.constant_init(m.bias, value=0)
        if len(model.sublayers()) == 0:
            m = model
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight, fan_in='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1)
                param_init.constant_init(m.bias, value=0)
            elif isinstance(m, nn.Linear):
                param_init.normal_init(m.weight, mean=0.0, std=0.01)
                param_init.constant_init(m.bias, value=0)


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    palette[-3:] = [255, 255, 255]
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Layer):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))
