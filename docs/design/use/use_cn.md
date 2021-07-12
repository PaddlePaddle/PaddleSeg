简体中文 | [English](use.md)
# 配置项

----
### train_dataset
* 训练数据集
>
>  * 参数
>     * type : 数据集类型，所支持值请参考[数据集文档](../../apis/datasets/datasets_cn.md)
>     * **others** : 请参考[对应模型的训练配置文件](../../../configs)

----
### val_dataset
* 验证数据集
>  * 参数
>     * type : 数据集类型，所支持值请参考[数据集文档](../../apis/datasets/datasets_cn.md)
>     * **others** : 请参考[对应模型的训练配置文件](../../../configs)
>

----
### batch_size
* 单张卡上，每步迭代训练时的数据量。一般来说，你所使用机器的显存越大，可以相应的调高batch_size的值。

----
### iters
* 使用一个 batch 数据对语义分割模型进行一次参数更新的过程称之为一次训练，即一次迭代。iters 即为训练过程中的迭代次数。

----
### optimizer
* 训练优化器
>  * 参数
>     * type : 优化器类型，目前只支持'sgd'和'adam'
>     * momentum : 动量优化法
>     * weight_decay : L2正则化的值

----
### lr_scheduler
* 学习率
>  * 参数
>     * type : 学习率类型，支持12种策略，分别是'PolynomialDecay', 'PiecewiseDecay', 'StepDecay', 'CosineAnnealingDecay', 'ExponentialDecay', 'InverseTimeDecay', 'LinearWarmup', 'MultiStepDecay', 'NaturalExpDecay', 'NoamDecay', ReduceOnPlateau, LambdaDecay.
>     * **others** : 请参考[Paddle官方LRScheduler文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

----
### learning_rate（不推荐使用该配置，将来会被废弃，建议使用`lr_scheduler`代替）
* 学习率
>  * 参数
>     * value : 初始学习率
>     * decay : 衰减配置
>       * type : 衰减类型，目前只支持poly
>       * power : 衰减率
>       * end_lr : 最终学习率

----
### loss
* 损失函数
>  * 参数
>     * types : 损失函数列表
>       * type : 损失函数类型，所支持值请参考损失函数库
>     * coef : 对应损失函数列表的系数列表

----
### model
* 待训练模型
>  * 参数
>     * type : 模型类型，所支持值请参考[模型库](../../apis/models/models_cn.md)
>     * **others** : 请参考[对应模型的训练配置文件](../../../configs)
---
### export
* 模型导出配置
>  * 参数
>    * transforms : 预测时的预处理操作，支持配置的transforms与`train_dataset`、`val_dataset`等相同。如果不填写该项，默认只会对数据进行归一化标准化操作。

# 示例

```yaml
batch_size: 4 # 设定迭代一次送入网络的图片数量。一般来说，你所使用机器的显存越大，可以调高batch_size的值。
iters: 80000 # 迭代次数

train_dataset: # 训练数据集
  type: Cityscapes # 训练数据集类的名称
  dataset_root: data/cityscapes # 训练数据集存放的目录
  transforms: # 数据变换与数据增强
    - type: ResizeStepScaling # 对图像按照某一个比例进行缩放，这个比例以scale_step_size为步长
      min_scale_factor: 0.5 # 缩放过程中涉及的参数
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop # 对图像和标注图进行随机裁剪
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip # 以一定的概率对图像进行水平翻转
    - type: Normalize # 对图像进行标准化
  mode: train # 训练模式

val_dataset: # 验证数据集
  type: Cityscapes # 验证数据集类的名称。
  dataset_root: data/cityscapes # 验证数据集存放的目录
  transforms:
    - type: Normalize # 对图像进行标准化
  mode: val # 验证模式

optimizer: # 使用何种优化器
  type: sgd # 随机梯度下降
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler: # 学习率的相关设置
  type: PolynomialDecay # 一种学习率类型。共支持12种策略
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss: # 使用何种损失函数
  types:
    - type: CrossEntropyLoss # 交叉熵损失函数
  coef: [1] #当使用了多种损失函数，可在 coef 中为每种损失指定配比

model: # 使用何种语义分割模型
  type: FCN
  backbone: # 使用何种骨干网络
    type: HRNet_W18
    pretrained: pretrained_model/hrnet_w18_ssld #指定预训练模型的存储路径
  num_classes: 19 # 像素类别数
  pretrained: Null
  backbone_indices: [-1]

```
