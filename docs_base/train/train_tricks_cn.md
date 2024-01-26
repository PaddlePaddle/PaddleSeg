简体中文 | [English](train_tricks.md)

# 模型训练技巧

## 损失函数使用权重

损失函数中，针对不同类别使用不同的权重，可以有效解决数据类别不均衡的问题。

语义分割常见的损失函数，比如CrossEntropyLoss和DiceLoss，都支持设置权重。

举例如下，如果背景和前景像素分别标注为0和1，则可以在`CrossEntropyLoss`字段中设置`weight`，分别表示对应下标类别的权重大小。注意，`weight`的长度需要等于类别数。

```
loss:
  types:
    - type: CrossEntropyLoss
      weight: [0.2, 0.8]
  coef: [1]
```

## 模型Backbone和Head使用不同的学习率

很多分割模型Backbone是加载大规模数据集上预训练的权重，所以Backbone模块的学习率可以比Head模块的学习率更小一些。

在`optimizer`配置字段中设置`backbone_lr_mult`，可以设置模型Backbone和Head使用不同学习率。

举例如下，`backbone`模块的学习率是`learning_rate * backbone_lr_mult`，其他模块的学习率是`learning_rate`。

```
optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5
  backbone_lr_mult: 0.1

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  end_lr: 0
  power: 0.9
```

## 线性学习率热身Warmup

线性学习率热身(Warmup)是对学习率进行初步调整，在正常调整学习率之前，先从小逐步增大学习率。

在`lr_scheduler`配置字段中设置`warmup_iters`和`warmup_start_lr`，开启线性学习率热身Warmup。

`warmup_iters`表示Warmup的轮数，`warmup_start_lr`表示最开始学习率，更多信息请参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LinearWarmup_cn.html#linearwarmup)。

```
lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  end_lr: 0
  power: 0.9
  warmup_iters: 1500
  warmup_start_lr: 1.0e-6
```
