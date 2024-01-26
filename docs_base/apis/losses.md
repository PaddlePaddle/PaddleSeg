# [paddleseg.models.losses](../../paddleseg/models/losses)

## LovaszSoftmaxLoss
> CLASS paddleseg.models.losses.LovaszSoftmaxLoss(ignore_index=255, classes='present')

    Multi-class Lovasz-Softmax loss.

> > Args
> > > - **ignore_index** (int64): Specifies a target value that is ignored and does not contribute to the input gradient. Default ``255``.
> > > - **classes** (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.


## LovaszHingeLoss
> CLASS paddleseg.models.losses.LovaszHingeLoss(ignore_index=255)

    Binary Lovasz hinge loss.

> > Args
> > > - **ignore_index** (int64): Specifies a target value that is ignored and does not contribute to the input gradient. Default ``255``.


## MixedLoss
> CLASS paddleseg.models.losses.MixedLoss(losses, coef)

    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.

> > Args
> > > - **losses** (list of nn.Layer): A list consisting of multiple loss classes
> > > - **coef** (float|int): Weighting coefficient of multiple loss

> > Returns
> > > - A callable object of MixedLoss.
