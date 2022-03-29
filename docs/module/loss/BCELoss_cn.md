简体中文 | [English](BCELoss_en.md)
# [BCELoss](../../../paddleseg/models/losses/binary_cross_entropy_loss.py)


二元交叉熵适合处理二分类与多标签分类任务。二元交叉熵以标注图的概率模型为基准，用二分类语义分割模型计算KL散度，根据吉布斯不等式知二者的交叉熵大于语义分割概率模型的熵。计算BCELoss时，我们通常忽略语义分割概率模型的熵（因为它是常量），仅将KL散度的一部分作为损失函数。


```python
class paddleseg.models.losses.BCELoss(
            weight = None,
            pos_weight = None,
            ignore_index = 255,
            edge_label = False
)
```

## BCELoss 使用指南


### 参数
* **weight**  (Tensor | str, optional): 对每个批数据元素的损失手动地重新调整权重。如果设定该参数，
且若传入的是一个 1D 张量，则其尺寸为 `[N, ]`，其数据类型为 float32 或 float64；
若传入的是一个 str，则值必须指定为 'dynamic'，以使在每轮迭代中根据二元交叉熵动态的计算权重。
            *默认:``'None'``*
* **pos_weight** (float|str, optional): 正样本的权重。若传入的是一个 str，则值必须指定为 'dynamic'，以使在每轮迭代中动态的计算权重。
            *默认:``'None'``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **edge_label** (bool, optional): 是否使用边缘标签。 *默认:``False``*
