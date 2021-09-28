简体中文 | [English](MixedLoss_en.md)
## [MixedLoss](../../../paddleseg/models/losses/mixed_loss.py)

实现混合loss训练。PaddleSeg每一种损失函数对应网络的一个logit 输出，如果要某个网络输出应用多种损失函数需要修改网络代码。MixedLoss 将允许网络对多个损失函数结果进行加权计算，只需以模块化的形式装入，就可以实现混合loss训练。

```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```


## Mixed loss使用指南

### 参数
* **losses** (list of nn.Layer): 由多个损失函数类所组成的列表。
* **coef** (float|int): 每个损失函数类的权重比。

### 返回值
* MixedLoss 类的可调用对象。
