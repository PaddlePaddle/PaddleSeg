简体中文 | [English](CrossEntropyLoss_en.md)
## [CrossEntropyLoss](../../../paddleseg/models/losses/cross_entropy_loss.py)


交叉熵 (CE) 方法由于其简单性和有效性，允许调整不同类别像素的权重，成为了一种最流行的损失函数。在很多语义分割任务中，交叉熵依赖于足够多的目标函数调用来准确估计基础分布的最佳参数。
CrossEntropyLoss常用于多像素类别的分割任务，其描述的是两个概率分布之间的不同，可以用来刻画当前模型与实际模型之间的差距（在训练过程中，我们暂时认为给出的标注集就是真实世界中的模型）。注意：机器学习算法中的逻辑回归是这种交叉熵的特例。
```python
class paddleseg.models.losses.CrossEntropyLoss(
                weight = None, 
                ignore_index = 255, 
                top_k_percent_pixels = 1.0
)
```

## Cross entropy loss 使用指南

### 参数
* **weight**  (tuple|list|ndarray|Tensor, optional): 为每个像素类别的损失手动调整权重。它的长度必须等同于像素类别数。可在多类样本不均衡等情况下调整各类的权重。
            *默认 ``None``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **top_k_percent_pixels** (float, optional): 该值的取值范围为 [0.0, 1.0]。 当该值 < 1.0 时，仅计算前 k% 像素（例如，前 20% 像素）的损失。 这将有助于对难分像素的挖掘。