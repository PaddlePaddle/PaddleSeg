简体中文 | [English](BootstrappedCrossEntropyLoss_en.md)
## [BootstrappedCrossEntropyLoss](../../../paddleseg/models/losses/bootstrapped_cross_entropy.py)
Bootstrapped 首先利用样本构造初始的分类起，然后对未标记样本进行迭代分类，进而利用扩展后的训练数据为未标记样本提取新的 seed rules。
[参考文献](https://arxiv.org/pdf/1412.6596.pdf)
```python
class paddleseg.models.losses.BootstrappedCrossEntropyLoss(
                        min_K, 
                        loss_th, 
                        weight = None, 
                        ignore_index = 255
)
```

## Bootstrapped cross entropy loss 使用指南

### 参数
* **min_K**  (int): 在计算损失时，参与计算的最小像素数。
* **loss_th** (float): 损失阈值。 只计算大于阈值的损失。
* **weight** (tuple|list, optional): 不同类的权重。 *默认:``None``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。 *默认:``255``*