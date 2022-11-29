简体中文 | [English](pre_config.md)

# 准备配置文件

PaddleSeg的配置文件按照模块化进行定义，包括超参、训练数据集、验证数据集、优化器、损失函数、模型等模块信息。

不同模块信息都对应PaddleSeg中定义的模块类，所以PaddleSeg基于配置文件构建对应的模块，进行模型训练、评估和导出。

PaddleSeg中所有语义分割模型都针对公开数据集，提供了对应的配置文件，保存在`PaddleSeg/configs`目录下。

下面是`PaddleSeg/configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml`配置文件。我们以这个配置文件为例进行详细解读，让大家熟悉修改配置文件的方法。

## 详细解读

超参主要包括batch_size和iters，前者是单卡的batch_size，后者表示训练迭代的轮数（单个batch进行一次前向和反向表示一轮）。

每个模块信息中，`type`字段对应到PaddleSeg代码中的模块类名(python class name)，其他字段对应模块类`__init__`函数的初始化参数。所以大家需要参考PaddleSeg代码中的模块类来修改模块信息。

数据集dataset模块，支持的dataset类在`PaddleSeg/paddleseg/datasets`[目录](../../paddleseg/datasets/)下，使用`@manager.DATASETS.add_component`进行注册。

数据预处理方式transforms模块，支持的transform类在`PaddleSeg/paddleseg/transforms/transforms.py`[文件](../../paddleseg/transforms/transforms.py)中，使用`@manager.TRANSFORMS.add_component`进行注册。

优化器optimizer模块，支持Paddle提供的所有优化器类，具体参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#api)。

学习率衰减lr_scheduler模块，支持Paddle提供的所有lr_scheduler类，具体参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#about-lr)。

损失函数Loss模块，在`types`字段下分别定义使用的损失函数类，`coef`字段定义每个损失函数的权重。`types`字段下损失函数个数，应该等于`coef`字段数组的长度。如果所有损失函数相同，可以只定义一个损失函数。支持的损失函数类在`PaddleSeg/paddleseg/models/losses/`[目录](../../paddleseg/models/losses/)下，使用`@manager.LOSSES.add_component注册`。

模型Model模块，支持的model类在`PaddleSeg/paddleseg/models/`[目录](../../paddleseg/models)下，使用`@manager.MODELS.add_component`注册。

模型Model模块，支持的backbone类在`PaddleSeg/paddleseg/models/backbones`[目录](../../paddleseg/models/backbones/)下，使用`@manager.BACKBONES.add_component`注册。

## 配置文件示例

```
batch_size: 4  #设定batch_size的值即为迭代一次送入网络的图片数量，一般显卡显存越大，batch_size的值可以越大。如果使用多卡训练，总得batch size等于该batch size乘以卡数。
iters: 1000    #模型训练迭代的轮数

train_dataset:  #训练数据设置
  type: Dataset #指定加载数据集的类。数据集类的代码在`PaddleSeg/paddleseg/datasets`目录下。
  dataset_root: data/optic_disc_seg #数据集路径
  train_path: data/optic_disc_seg/train_list.txt  #数据集中用于训练的标识文件
  num_classes: 2  #指定类别个数（背景也算为一类）
  mode: train #表示用于训练
  transforms: #模型训练的数据预处理方式。
    - type: ResizeStepScaling #将原始图像和标注图像随机缩放为0.5~2.0倍
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop #从原始图像和标注图像中随机裁剪512x512大小
      crop_size: [512, 512]
    - type: RandomHorizontalFlip  #对原始图像和标注图像随机进行水平反转
    - type: RandomDistort #对原始图像进行亮度、对比度、饱和度随机变动，标注图像不变
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize #对原始图像进行归一化，标注图像保持不变

val_dataset:  #验证数据设置
  type: Dataset #指定加载数据集的类。数据集类的代码在`PaddleSeg/paddleseg/datasets`目录下。
  dataset_root: data/optic_disc_seg #数据集路径
  val_path: data/optic_disc_seg/val_list.txt  #数据集中用于验证的标识文件
  num_classes: 2  #指定类别个数（背景也算为一类）
  mode: val #表示用于验证
  transforms: #模型验证的数据预处理的方式
    - type: Normalize #对原始图像进行归一化，标注图像保持不变

optimizer: #设定优化器的类型
  type: sgd #采用SGD（Stochastic Gradient Descent）随机梯度下降方法为优化器
  momentum: 0.9 #设置SGD的动量
  weight_decay: 4.0e-5 #权值衰减，使用的目的是防止过拟合

lr_scheduler: # 学习率的相关设置
  type: PolynomialDecay # 一种学习率类型。共支持12种策略
  learning_rate: 0.01 # 初始学习率
  power: 0.9
  end_lr: 0

loss: #设定损失函数的类型
  types:
    - type: CrossEntropyLoss  #CE损失
  coef: [1, 1, 1] # PP-LiteSeg有一个主loss和两个辅助loss，coef表示权重，所以 total_loss = coef_1 * loss_1 + .... + coef_n * loss_n

model:  #模型说明
  type: PPLiteSeg  #设定模型类别
  backbone:  # 设定模型的backbone，包括名字和预训练权重
    type: STDC2
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz

```

## 其他

注意：
- 对于训练和测试数据集的预处理，PaddleSeg默认会添加读取图像操作、HWC转CHW的操作，所以这两个操作不用添加到transform配置字段中。
- 只有"PaddleSeg/configs/quick_start"下面配置文件中的学习率为单卡学习率，其他配置文件中均为4卡的学习率。如果大家单卡训练来复现公开数据集上的指标，学习率设置应变成原来的1/4。


上面我们介绍的PP-LiteSeg配置文件，所有的配置信息都放置在同一个yml文件中。为了具有更好的复用性，PaddleSeg的配置文件采用了更加耦合的设计，配置文件支持包含复用。

如下图，右侧`deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml`配置文件通过`_base_: '../_base_/cityscapes.yml'`来包含左侧`cityscapes.yml`配置文件，其中`_base_: `设置的是被包含配置文件相对于该配置文件的路径。

如果两个配置文件具有相同的字段信息，被包含的配置文件中的字段信息会被覆盖。如下图，1号配置文件可以覆盖2号配置文件的字段信息。

![](./images/fig3.png)
