简体中文 | [English](EdgeAttentionLoss_en.md)
## [EdgeAttentionLoss](../../../paddleseg/models/losses/edge_attention_loss.py)
适合以 encoder 提取edge，以 decoder 进行加权聚合的多任务训练场景。是一种融合边缘检测与注意力机制进行多 loss 的组合输出的方法。

```python
class paddleseg.models.losses.EdgeAttentionLoss(
                edge_threshold = 0.8, 
                ignore_index = 255
)
```

## Edge attention loss 使用指南

### 参数
* **edge_threshold** (float): 值大于 edge_threshold 的像素被视为边缘。
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*