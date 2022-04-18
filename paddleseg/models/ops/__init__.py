import os
from paddle.autograd import PyLayer
from paddle.utils.cpp_extension import load

__all__ = ['psa_mask_op']

custom_ops = load(
    name="psamask",
    sources=[
        "paddleseg/models/ops/psamask.cc", "paddleseg/models/ops/psamask.cu"
    ], )
psa_mask_op = custom_ops.psamask
