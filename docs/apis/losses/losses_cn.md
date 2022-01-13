简体中文 | [English](losses.md)
# [paddleseg.models.losses](../../../paddleseg/models/losses)


## [BCELoss](../../../paddleseg/models/losses/binary_cross_entropy_loss.py)
```python
class paddleseg.models.losses.BCELoss(
            weight = None,
            pos_weight = None,
            ignore_index = 255,
            edge_label = False
)
```

> Binary cross entropy 损失函数的实现。

### 参数
* **weight**  (Tensor | str, optional): 对每个批数据元素的损失手动地重新调整权重。如果设定该参数，
且若传入的是一个 1D 张量，则其尺寸为 `[N, ]`，其数据类型为 float32 或 float64；
若传入的是一个 str，则值必须指定为 'dynamic'，以使在每轮迭代中根据二元交叉熵动态的计算权重。
            *默认:``'None'``*
* **pos_weight** (float|str, optional): 正样本的权重。若传入的是一个 str，则值必须指定为 'dynamic'，以使在每轮迭代中动态的计算权重。
            *默认:``'None'``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **edge_label** (bool, optional): 是否使用边缘标签。 *默认:``False``*



## [BootstrappedCrossEntropyLoss](../../../paddleseg/models/losses/bootstrapped_cross_entropy.py)
```python
class paddleseg.models.losses.BootstrappedCrossEntropyLoss(
                        min_K, 
                        loss_th, 
                        weight = None, 
                        ignore_index = 255
)
```
> Bootstrapped cross entropy 损失函数的实现。

### 参数
* **min_K**  (int): 在计算损失时，参与计算的最小像素数。
* **loss_th** (float): 损失阈值。 只计算大于阈值的损失。
* **weight** (tuple|list, optional): 不同类的权重。 *默认:``None``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。 *默认:``255``*




## [CrossEntropyLoss](../../../paddleseg/models/losses/cross_entropy_loss.py)
```python
class paddleseg.models.losses.CrossEntropyLoss(
                weight = None, 
                ignore_index = 255, 
                top_k_percent_pixels = 1.0
)
```

> Cross entropy 损失函数的实现。

### 参数
* **weight**  (tuple|list|ndarray|Tensor, optional): 为每个像素类别的损失手动调整权重。它的长度必须等同于像素类别数。可在多类样本不均衡等情况下调整各类的权重。
            *默认 ``None``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **top_k_percent_pixels** (float, optional): 该值的取值范围为 [0.0, 1.0]。 当该值 < 1.0 时，仅计算前 k% 像素（例如，前 20% 像素）的损失。 这将有助于对难分像素的挖掘。



## [RelaxBoundaryLoss](../../../paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py)
```python
class paddleseg.models.losses.RelaxBoundaryLoss(
                border = 1,
                calculate_weights = False,
                upper_bound = 1.0,
                ignore_index = 255
)
```

> Relax boundary 损失函数的实现。

### 参数
* **border**  (int, optional): 边界的松弛值。*默认:``1``*
* **calculate_weights** (bool, optional): 是否计算所有类别的权重。 *默认:``False``*
* **upper_bound** (float, optional): 如果为所有类别计算权重，则指定权重的上限值。 *默认:``1.0``*
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*




## [DiceLoss](../../../paddleseg/models/losses/dice_loss.py)
```python
class paddleseg.models.losses.DiceLoss(
            ignore_index = 255, 
            smooth = 0.
)
```

> Dice 损失函数的实现。

### 参数
* **ignore_index**  (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **smooth** (float, optional): 可以添加该smooth参数以防止出现除0异常。你也可以设置更大的平滑值（拉普拉斯平滑）以避免过拟合。*默认:`` 0``*



## [EdgeAttentionLoss](../../../paddleseg/models/losses/edge_attention_loss.py)
```python
class paddleseg.models.losses.EdgeAttentionLoss(
                edge_threshold = 0.8, 
                ignore_index = 255
)
```

> Edge attention 损失函数的实现。

### 参数
* **edge_threshold** (float): 值大于 edge_threshold 的像素被视为边缘。
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*



## [DualTaskLoss](../../../paddleseg/models/losses/gscnn_dual_task_loss.py)
```python
class paddleseg.models.losses.DualTaskLoss(
            ignore_index = 255, 
            tau = 0.5
)
```

> Dual task 损失函数的实现。

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **tau** (float): Gumbel softmax 样本的tau。


## [L1Loss](../../../paddleseg/models/losses/l1_loss.py)
```python
class paddleseg.models.losses.L1Loss(
            reduction = 'mean', 
            ignore_index = 255
)
```

> L1 损失函数的实现。

### 参数
* **reduction** (str, optional): 指示应用于损失值的 reduction 方式，可以指定为 ``'none'`` 或 ``'none'`` 或``'sum'``。

    > - 如果 `reduction` 为 ``'none'``， 不对损失值做任何处理直接返回；
    > - 如果 `reduction` 为 ``'mean'``， 返回经 Mean 处理后的损失；
    > - 如果 `reduction` 为 ``'sum'``， 返回经 Sum 处理后的损失。
    > - *默认:``'mean'``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*




## [MSELoss](../../../paddleseg/models/losses/mean_square_error_loss.py)
```python
class paddleseg.models.losses.MSELoss(
            reduction = 'mean', 
            ignore_index = 255
)
```

> Mean square error 损失函数的实现。

### 参数
* **reduction** (string, optional): 对输出结果的 reduction 方式，可以指定为 ``'none'`` 或 ``'none'`` 或``'sum'``。

    > - 如果 `reduction` 为 ``'none'``， 不对损失值做任何处理直接返回；
    > - 如果 `reduction` 为 ``'mean'``， 返回经 Mean 处理后的损失；
    > - 如果 `reduction` 为 ``'sum'``， 返回经 Sum 处理后的损失。
    > - *默认:``'mean'``*
* **ignore_index** (int, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*



## [OhemCrossEntropyLoss](../../../paddleseg/models/losses/ohem_cross_entropy_loss.py)
```python
class paddleseg.models.losses.OhemCrossEntropyLoss(
                thresh = 0.7, 
                min_kept = 10000, 
                ignore_index = 255
)
```

> Ohem cross entropy 损失函数的实现。

### 参数
* **thresh** (float, optional): ohem的阈值。 *默认:``0.7``*
* **min_kept** (int, optional): 指定最小保持用于计算损失函数的像素数。``min_kept`` 与 ``thresh`` 配合使用：如果 ``thresh`` 设置过高，可能导致本轮迭代中没有对损失函数的输入值，因此设定该值可以保证至少前``min_kept``个元素不会被过滤掉。*默认:``10000``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*



## [OhemEdgeAttentionLoss](../../../paddleseg/models/losses/ohem_edge_attention_loss.py)
```python
class paddleseg.models.losses.OhemEdgeAttentionLoss(
                edge_threshold = 0.8,
                thresh = 0.7,
                min_kept = 5000,
                ignore_index = 255
)
```

> Ohem edge attention 损失函数的实现。

### 参数
* **edge_threshold** (float, optional): 值大于 edge_threshold 的像素被视为边缘。 *默认:``0.8``*
* **thresh** (float, optional): ohem的阈值。 *默认:`` 0.7``*
* **min_kept** (int, optional): 指定最小保持用于计算损失函数的像素数。``min_kept`` 与 ``thresh`` 配合使用：如果 ``thresh`` 设置过高，可能导致本轮迭代中没有对损失函数的输入值，因此设定该值可以保证至少前``min_kept``个元素不会被过滤掉。*默认:``5000``*
* **ignore_index** (int64, optional): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*


## [LovaszSoftmaxLoss](../../../paddleseg/models/losses/lovasz_loss.py)
```python
class paddleseg.models.losses.LovaszSoftmaxLoss(
            ignore_index = 255, 
            classes = 'present'
)
```

> 多类别 Lovasz-Softmax 损失函数的实现。

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **classes** (str|list): 'all' 表示所有，'present' 表示标签中存在的类，或者要做 average 的类列表。


## [LovaszHingeLoss](../../../paddleseg/models/lovasz_loss.py)
```python
class paddleseg.models.losses.LovaszHingeLoss(ignore_index = 255)
```

> Binary Lovasz hinge 损失函数的实现。

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*


## [MixedLoss](../../../paddleseg/models/mixed_loss.py)
```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```

> 对多个损失函数结果的加权计算。
> 其优点为在不改变网络代码的情况下，实现混合损失训练。

### 参数
* **losses** (list of nn.Layer): 由多个损失函数类所组成的列表。
* **coef** (float|int): 每个损失函数类的权重比。

### 返回值
* MixedLoss 类的可调用对象。
