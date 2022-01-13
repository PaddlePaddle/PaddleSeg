简体中文 | [English](DualTaskLoss_en.md)
## [DualTaskLoss](../../../paddleseg/models/losses/gscnn_dual_task_loss.py)
用于为半监督学习的 Dual-task 一致性以对模型进行约束。DualTaskLoss 旨在强化多个任务之间的一致性。

```python
class paddleseg.models.losses.DualTaskLoss(
            ignore_index = 255, 
            tau = 0.5
)
```

## Dual task  loss 使用指南

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **tau** (float): Gumbel softmax 样本的tau。