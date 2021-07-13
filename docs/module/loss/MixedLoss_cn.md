## [MixedLoss](../../../paddleseg/models/mixed_loss.py)
```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```

> 对多个损失函数结果的加权计算。
> 其优点为在不改变网络代码的情况下，实现混合损失训练。

## Mixed loss使用指南

### 参数
* **losses** (list of nn.Layer): 由多个损失函数类所组成的列表。
* **coef** (float|int): 每个损失函数类的权重比。

### 返回值
* MixedLoss 类的可调用对象。