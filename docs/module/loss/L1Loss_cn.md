简体中文 | [English](L1Loss_en.md)
## [L1Loss](../../../paddleseg/models/losses/l1_loss.py)
L1范数损失函数用于计算最小绝对值偏差。该损失旨在将估计值与真实值之间的绝对差值的总和最小化。可以选择配合使用 reduction 策略对该 loss 的直接计算结果进行一定的处理。

```python
class paddleseg.models.losses.L1Loss(
            reduction = 'mean', 
            ignore_index = 255
)
```

## L1  loss 使用指南

### 参数
* **reduction** (str, optional): 指示应用于损失值的 reduction 方式，可以指定为 ``'none'`` 或 ``'none'`` 或``'sum'``。

    > - 如果 `reduction` 为 ``'none'``， 不对损失值做任何处理直接返回；
    > - 如果 `reduction` 为 ``'mean'``， 返回经 Mean 处理后的损失；
    > - 如果 `reduction` 为 ``'sum'``， 返回经 Sum 处理后的损失。
    > - *默认:``'mean'``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*