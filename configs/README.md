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
>     * type : 优化器类型，目前只支持sgd
>     * momentum : 动量
>     * weight_decay : L2正则化的值

----
### learning_rate
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
>     * coef : 对应损失函数列表的系数列表

----
### model
> 待训练模型
>  * 参数
>     * type : 模型类型，所支持值请参考模型库
>     * **others** : 请参考对应模型训练配置文件


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

learning_rate:
  value: 0.01
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0

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
