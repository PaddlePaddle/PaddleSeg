English | [简体中文](MixedLoss_cn.md)
## [MixedLoss](../../../paddleseg/models/mixed_loss.py)
```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```

> Weighted calculation of multiple loss function results.
> Its advantage is to realize mixed loss training without changing the network code.

## Mixed loss usage guidance

### Args
* **losses** (list of nn.Layer): A list consisting of multiple loss classes
* **coef** (float|int): Weighting coefficient of multiple loss

### Returns
* A callable object of MixedLoss.