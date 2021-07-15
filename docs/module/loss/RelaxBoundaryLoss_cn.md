简体中文 | [English](RelaxBoundaryLoss_en.md)
## [RelaxBoundaryLoss](../../../paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py)
为了提升分割效果，对边界的处理至关重要。通常使用神经网络预测边界图。在这种训练过程中，通常以边界为划分依据，将对边界敏感的像素划分为若干类别，最后分别计算交叉熵并将其作为损失函数输出。
```python
class paddleseg.models.losses.RelaxBoundaryLoss(
                border = 1,
                calculate_weights = False,
                upper_bound = 1.0,
                ignore_index = 255
)
```

## Relax boundary loss使用指南

### 参数
* **border**  (int, optional): 边界的松弛值。*默认:``1``*
* **calculate_weights** (bool, optional): 是否计算所有类别的权重。 *默认:``False``*
* **upper_bound** (float, optional): 如果为所有类别计算权重，则指定权重的上限值。 *默认:``1.0``*
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*