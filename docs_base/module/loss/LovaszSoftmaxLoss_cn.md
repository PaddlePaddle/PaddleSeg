简体中文 | [English](LovaszSoftmaxLoss_en.md)
## [LovaszSoftmaxLoss](../../../paddleseg/models/losses/lovasz_loss.py)

lovasz softmax loss适用于多分类问题。该工作发表在CVPR 2018上。
[参考文献](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html)

```python
class paddleseg.models.losses.LovaszSoftmaxLoss(
            ignore_index = 255,
            classes = 'present'
)
```

## Lovasz-Softmax loss使用指南

### 参数
* **ignore_index** (int64): 指定一个在标注图中要忽略的像素值，其对输入梯度不产生贡献。当标注图中存在无法标注（或很难标注）的像素时，可以将其标注为某特定灰度值。在计算损失值时，其与原图像对应位置的像素将不作为损失函数的自变量。 *默认:``255``*
* **classes** (str|list): 'all' 表示所有，'present' 表示标签中存在的类，或者要做 average 的类列表。
