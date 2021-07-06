import paddle
import paddle.nn as nn
import numpy as np
from paddle.fluid.initializer import  Initializer

class SInitializer(Initializer):
    def __init__(self, local_init=True, gamma=None):
        self.local_init = local_init
        self.gamma = gamma

    def __call__(self, m):


        if isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D,
                          nn.InstanceNorm1D, nn.InstanceNorm2D, nn.InstanceNorm3D,
                          nn.GroupNorm, nn.SyncBatchNorm)) or 'BatchNorm' in m.__class__.__name__:
            if m.weight is not None:
                self._init_gamma(m.weight)
            if m.bias is not None:
                self._init_beta(m.bias)
            else:
                if getattr(m, 'weight', None) is not None:
                    self._init_weight(m.weight)
                if getattr(m, 'bias', None) is not None:
                    self._init_bias(m.bias)

    def _init_weight(self, param):
        initializer = nn.initializer.Uniform(-0.07, 0.07)
        initializer(param, param.block)

    def _init_bias(self, param):
        initializer = nn.initializer.Constant(0)
        initializer(param, param.block)

    def _init_gamma(self, param):
        if self.gamma is None:
            initializer = nn.initializer.Constant(0)
            initializer(param, param.block)
        else:
            initializer = nn.initializer.Normal(1, self.gamma)
            initializer(param, param.block)

    def _init_beta(self, param):
        initializer = nn.initializer.Constant(0)
        initializer(param, param.block)

        
class XavierGluon(SInitializer):
    def __init__(self, rnd_type='uniform', factor_type='avg', magnitude=3, **kwargs):
        super().__init__(**kwargs)

        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, arr):
        fan_in, fan_out =self._compute_fans(arr)

        if self.factor_type == 'avg':
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == 'in':
            factor = fan_in
        elif self.factor_type == 'out':
            factor = fan_out
        else:
            raise ValueError('Incorrect factor type')
        scale = np.sqrt(self.magnitude / factor)

        if self.rnd_type == 'uniform':
            initializer = nn.initializer.Uniform(-scale, scale)
            initializer(arr, arr.block)
        elif self.rnd_type == 'gaussian':
            initializer = nn.initializer.Normal(0, scale)
            initializer(arr, arr.block)
        else:
            raise ValueError('Unknown random type')
