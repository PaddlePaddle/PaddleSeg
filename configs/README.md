# 配置项

----
### train_dataset
>  训练数据集
>
>  * 参数
>     * type : 数据集类型，所支持值请参考训练配置文件
>     * **others** : 请参考对应模型训练配置文件

----
### val_dataset
>  评估数据集
>  * 参数
>     * type : 数据集类型，所支持值请参考训练配置文件
>     * **others** : 请参考对应模型训练配置文件
>

----
### batch_size
>  单张卡上，每步迭代训练时的数据量

----
### iters
>  训练步数

----
### optimizer
> 训练优化器
>  * 参数
>     * type : 优化器类型，目前只支持'sgd'和'adam'
>     * momentum : 动量
>     * weight_decay : L2正则化的值

----
### lr_scheduler
> 学习率
>  * 参数
>     * type : 学习率类型，支持10种策略，分别是'PolynomialDecay', 'PiecewiseDecay', 'StepDecay', 'CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'LinearWarmup', 'MultiStepDecay', 'NaturalExpDecay', 'NoamDecay'.
>     * **others** : 请参考[Paddle官方LRScheduler文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

----
### learning_rate（不推荐使用该配置，将来会被废弃，建议使用`lr_scheduler`代替）
> 学习率
>  * 参数
>     * value : 初始学习率
>     * decay : 衰减配置
>       * type : 衰减类型，目前只支持poly
>       * power : 衰减率
>       * end_lr : 最终学习率

----
### loss
> 损失函数
>  * 参数
>     * types : 损失函数列表
>       * type : 损失函数类型，所支持值请参考损失函数库
>       * ignore_index : 训练过程需要忽略的类别，默认取值与`train_dataset`的ignore_index一致，**推荐不用设置此项**。如果设置了此项，`loss`和`train_dataset`的ignore_index必须相同。
>     * coef : 对应损失函数列表的系数列表

----
### model
> 待训练模型
>  * 参数
>     * type : 模型类型，所支持值请参考模型库
>     * **others** : 请参考对应模型训练配置文件
---
### export
> 模型导出配置
>  * 参数
>    * transforms : 预测时的预处理操作，支持配置的transforms与`train_dataset`、`val_dataset`等相同。如果不填写该项，默认只会对数据进行归一化标准化操作。

# 示例

```yaml
batch_size: 4
iters: 80000

train_dataset:
  type: Cityscapes
  dataset_root: data/cityscapes
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip
    - type: Normalize
  mode: train

val_dataset:
  type: Cityscapes
  dataset_root: data/cityscapes
  transforms:
    - type: Normalize
  mode: val

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]

model:
  type: FCN
  backbone:
    type: HRNet_W18
    pretrained: pretrained_model/hrnet_w18_ssld
  num_classes: 19
  pretrained: Null
  backbone_indices: [-1]

```
