简体中文 | [English](OhemEdgeAttentionLoss_en.md)
## [OhemEdgeAttentionLoss](../../../paddleseg/models/losses/ohem_edge_attention_loss.py)
OHEM算法将根据输入到模型中的样本的损失来区分出困难样本，这些困难样本分类精度差，会产生较大的损失。在存在困难样本的场景下，如欲提高提取边缘的性能，可以使用该损失函数。
```python
class paddleseg.models.losses.OhemEdgeAttentionLoss(
                edge_threshold = 0.8,
                thresh = 0.7,
                min_kept = 5000,
                ignore_index = 255
)
```

## Ohem edge attention loss使用指南

### 参数
* **edge_threshold** (float, optional): 值大于 edge_threshold 的像素被视为边缘。 *默认:``0.8``*
* **thresh** (float, optional): ohem的阈值。 *默认:`` 0.7``*
* **min_kept** (int, optional): 指定最小保持用于计算损失函数的像素数。``min_kept`` 与 ``thresh`` 配合使用：如果 ``thresh`` 设置过高，可能导致本轮迭代中没有对损失函数的输入值，因此设定该值可以保证至少前``min_kept``个元素不会被过滤掉。*默认:``5000``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
