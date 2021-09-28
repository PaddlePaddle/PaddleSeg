简体中文 | [English](OhemCrossEntropyLoss_en.md)
## [OhemCrossEntropyLoss](../../../paddleseg/models/losses/ohem_cross_entropy_loss.py)
OHEM旨在解决处理困难样本的问题。在一些语义分割问题中，经常出现像素点难以标注或无法标注的情况，或是类别不平衡的情况，都将对模型性能产生严重的制约。OHEM算法将根据输入到模型中的样本的损失来区分出困难样本，这些困难样本分类精度差，会产生较大的损失。

```python
class paddleseg.models.losses.OhemCrossEntropyLoss(
                thresh = 0.7,
                min_kept = 10000,
                ignore_index = 255
)
```

## Ohem cross entropy loss使用指南

### 参数
* **thresh** (float, optional): ohem的阈值。 *默认:``0.7``*
* **min_kept** (int, optional): 指定最小保持用于计算损失函数的像素数。``min_kept`` 与 ``thresh`` 配合使用：如果 ``thresh`` 设置过高，可能导致本轮迭代中没有对损失函数的输入值，因此设定该值可以保证至少前``min_kept``个元素不会被过滤掉。*默认:``10000``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
