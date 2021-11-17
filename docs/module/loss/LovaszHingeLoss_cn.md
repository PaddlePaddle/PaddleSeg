简体中文 | [English](LovaszHingeLoss_en.md)
## [LovaszHingeLoss](../../../paddleseg/models/losses/lovasz_loss.py)
Hinge Loss是在不连续、不平滑的简单阶梯损失函数上改进的一种损失函数。对于正样本，Hinge Loss的输出应大于等于1；对于正样本，Hinge Loss的输出应小于等于-1。

```python
class paddleseg.models.losses.LovaszHingeLoss(ignore_index = 255)
```

## Binary Lovasz hinge loss使用指南

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
