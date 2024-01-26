简体中文 | [English](MSELoss_en.md)
## [MSELoss](../../../paddleseg/models/losses/mean_square_error_loss.py)
Mean square error loss 即均方根误差。均方根指模型预测值与样本真实值之间距离的平方的平均值。

```python
class paddleseg.models.losses.MSELoss(
            reduction = 'mean',
            ignore_index = 255
)
```

## Mean square error loss使用指南

### 参数
* **reduction** (string, optional): 对输出结果的 reduction 方式，可以指定为 ``'none'`` 或 ``'none'`` 或``'sum'``。

    > - 如果 `reduction` 为 ``'none'``， 不对损失值做任何处理直接返回；
    > - 如果 `reduction` 为 ``'mean'``， 返回经 Mean 处理后的损失；
    > - 如果 `reduction` 为 ``'sum'``， 返回经 Sum 处理后的损失。
    > - *默认:``'mean'``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
