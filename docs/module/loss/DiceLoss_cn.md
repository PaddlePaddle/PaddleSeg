简体中文 | [English](DiceLoss_en.md)
## [DiceLoss](../../../paddleseg/models/losses/dice_loss.py)
Dice Loss 是一种广泛的应用于医学影像分割任务中的损失函数。Dice 系数是一种用于度量集合之间的相似程度的函数，在语义分割任务中，我们可以理解为当前的模型与真实世界中的真实模型之间的相似程度。Dice损失函数的计算过程包括用预测分割图与GT分割图之间进行点乘、对点乘结果的每个位置进行累计求和，最后计算 1-Dice 的值作为损失函数的输出，即 Dice = 1-2(|X∩Y|/|X|+|Y|)。你可以使用拉普拉斯平滑系数，将分子分母添加该系数后，可以避免除0异常，同时减少过拟合。即Dice_smooth = 1-2((|X∩Y|+smooth) / (|X|+|Y|+smooth) )
```python。

class paddleseg.models.losses.DiceLoss(
            ignore_index = 255, 
            smooth = 0.
)
```

## Dice loss 使用指南

### 参数
* **ignore_index**  (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **smooth** (float, optional): 可以添加该smooth参数以防止出现除0异常。你也可以设置更大的平滑值（拉普拉斯平滑）以避免过拟合。*默认:`` 0``*