
class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult