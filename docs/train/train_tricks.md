English | [简体中文](train_tricks_cn.md)

# Training Tricks

## Weights for different classes in loss

Using weights for different classes in loss is an effective method to solve class imbalance problem.

The CrossEntropyLoss and DiceLoss in PaddleSeg support adding `weight` config.

For example, if the foreground and background are labeled as 0 and 1, we set the weight of foreground and background as 0.2 and 0.8, respectively.

```
loss:
  types:
    - type: CrossEntropyLoss
      weight: [0.2, 0.8]
  coef: [1]
```

## Different learning rate for backbone and head

Since many segmentation models load the pretrained weights for backbone, the learning rate of the backbone can be lower than that of other modules.

We only need to set `backbone_lr_mult` to the `optimizer` in config file as follows.
Then, the learning rate of backbone and other modules are `learning_rate * backbone_lr_mult` and `learning_rate`, respectively.

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

## Linear warmup learning scheduler

The linear warmup learning scheduler gradually increases the lerning rate for from a small value to the initial value before adjusting the learning rate normally.

Setting `warmup_iters` and `warmup_start_lr` in `lr_scheduler` config to enable  linear warmup learning scheduler.

`warmup_iters` denotes the iters for warmup, `warmup_start_lr` denotes the small value mentioned above. For more information, please refer to [doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LinearWarmup_cn.html#linearwarmup)

```
lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  end_lr: 0
  power: 0.9
  warmup_iters: 1500
  warmup_start_lr: 1.0e-6
```
