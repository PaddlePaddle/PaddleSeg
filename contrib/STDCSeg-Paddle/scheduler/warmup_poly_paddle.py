from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import PolynomialDecay


class PolyDecay(PolynomialDecay):
    def __init__(self, lr, power, decay_steps, end_lr,cycle=False,**kwargs):
        super(PolyDecay, self).__init__(
            learning_rate=lr,
            power = power,
            decay_steps = decay_steps,
            end_lr = end_lr,
            cycle = cycle
            )
        self.update_specified = False

class Warmup_PolyDecay(LinearWarmup):

    def __init__(self, lr_rate,warmup_steps,iters,warmup_strat_lr = 1e-5,end_lr=1e-5,power=0.9,cycle = False,**kwargs):
        assert iters > warmup_steps, "total epoch({}) should be larger than warmup_epoch({}) in WarmupPolyDecay.".format(iters, warmup_steps)
        warmup_steps = warmup_steps
        start_lr = warmup_strat_lr
        end_lr = end_lr
        warmup_end_lr = lr_rate
        lr_sch = PolyDecay(lr = lr_rate,end_lr=end_lr, decay_steps= iters-warmup_steps,power=power,cycle=cycle)

        super(Warmup_PolyDecay, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            end_lr=warmup_end_lr)

        self.update_specified = False