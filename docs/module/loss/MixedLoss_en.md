English | [简体中文](MixedLoss_cn.md)
## [MixedLoss](../../../paddleseg/models/losses/mixed_loss.py)

Realize mixed loss training. Each loss function of PaddleSeg corresponds to a logit output of the network. If you want to apply multiple loss functions to a certain network output, you need to modify the network code. MixedLoss will allow the network to weight the results of multiple loss functions, and only need to be loaded in a modular form to achieve mixed loss training.

```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```


## Mixed loss usage guidance

### Args
* **losses** (list of nn.Layer): A list consisting of multiple loss classes
* **coef** (float|int): Weighting coefficient of multiple loss

### Returns
* A callable object of MixedLoss.
