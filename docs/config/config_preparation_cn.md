# 准备配置文件
PaddleSeg提供**配置化驱动方式**进行模型训练、测试和预测等，配置文件是其中的关键。
## 配置文件的详细解读
PaddleSeg中，配置文件包括超参、训练数据集、验证数据集、优化器、损失函数、模型等信息。**所有的配置文件都在PaddleSeg/configs文件夹下面**。
  
大家可以灵活修改配置文件的内容，如自定义模型使用的骨干网络、模型使用的损失函数以及关于网络结构等配置，自定义配置数据处理的策略，如改变尺寸、归一化和翻转等数据增强的策略，这些修改可以参考对应模块的代码，传入相应参数即可。

### 配置文件示例
以```PaddleSeg/configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml```为例，详细解读配置文件如下:
```
batch_size: 4  #设定batch_size的值即为迭代一次送入网络的图片数量，一般显卡显存越大，batch_size的值可以越大。如果使用多卡训练，总得batch size等于该batch size乘以卡数。
iters: 1000    #模型训练迭代的轮数

train_dataset:  #训练数据设置
  type: Dataset #指定加载数据集的类
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
  type: Dataset #指定加载数据集的类
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

### 配置文件说明
* 对于训练和测试数据集的预处理，PaddleSeg默认会添加读取图像操作、HWC转CHW的操作，所以这两个操作不用添加到transform配置字段中。
* 只有"PaddleSeg/configs/quick_start"下面配置文件中的学习率为单卡学习率，其他配置文件中均为4卡的学习率。如果大家单卡训练来复现公开数据集上的指标，学习率设置应变成原来的1/4。

>超参设置
```
batch_size: 4 
iters: 1000
```
* 其中batch_size的值为迭代一次送入网络的图片数量，用户可以根据显卡显存情况进行调整，一般原则为显存越大，batch_size适当加大。
* 其中iters的值为模型迭代的轮数，用户可以自行调整。

>训练数据设置[（参数对应代码）](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/datasets/dataset.py)
```
train_dataset:  
  type: Dataset 
  dataset_root: data/optic_disc_seg 
  mode: train 
  train_path: data/optic_disc_seg/train_list.txt 
  num_classes: 2  
  transforms: 
    - type: ResizeStepScaling 
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop 
      crop_size: [512, 512]
    - type: RandomHorizontalFlip  
    - type: RandomDistort 
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize 
```
* 其中type是指定加载数据集的类，Dataset类位于```PaddleSeg/paddleseg/datasets/dataset.py```中。
* 其中dataset_root是数据集的路径，字符串类型，用户可根据自己的数据集存放位置对该参数进行调整。
* 其中mode是模式，字符串类型（非必填项），可以在(train, val, test)选择其一进行填写，默认情况下为train。
* 其中train_path是数据集中用于训练的标识文件，字符串类型（非必填项），当mode参数为train时，train_path这个参数则成为必填项，train_path file的内容例如:
```
image1.jpg ground_truth1.png
image2.jpg ground_truth2.png
```
类似的，还有val_path，test_path，当mode参数为val时，val_path这个参数则成为必填项，val_path file的内容和train_path file的内容类似。
* 其中separator是数据集列表的分割符，字符串类型（非必填项），默认按照空格进行分割。
* 其中edge表示训练时是否开启边缘计算，是布尔类型（非必填项），默认为False。
* num_classes：表示类别个数（背景也算一类），用户可根据实际情况进行自行调整。
* transforms：表示模型训练时的数据预处理方式[（数据预处理代码）](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/transforms/transforms.py)，是一个list，list中的内容可以填写：
```
# 将原始图像和标注图像随机缩放为0.5~2.0倍
ResizeStepScaling(min_scale_factor=0.5, max_scale_factor=0.2,scale_step_size=0.25),
# 从原始图像和标注图像中随机裁剪512x512大小
RandomPaddingCrop(crop_size=(512,512)),
# 对原始图像和标注图像随机进行水平反转
RandomHorizontalFlip(),
# 对原始图像进行亮度、对比度、饱和度随机变动，标注图像不变
RandomDistort(brightness_range=0.5, contrast_range=0.5, saturation_range=0.5),
# 对原始图像进行归一化，标注图像保持不变
Normalize()
```

>验证数据设置[参数对应代码](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/datasets/dataset.py)
```
val_dataset:  
  type: Dataset 
  dataset_root: data/optic_disc_seg 
  val_path: data/optic_disc_seg/val_list.txt  
  num_classes: 2  
  mode: val 
  transforms: 
    - type: Normalize 
```
* 其中验证数据集的设置与训练数据集的设置类似，此处不再赘述。

>优化器设置[参数对应代码](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/cvlibs/config.py)
```
optimizer: 
  type: sgd 
  momentum: 0.9 
  weight_decay: 4.0e-5 
```
* 其中type表示选用优化器的类型，可选SGD、Adam等，如果type选用SGD需要添加momentum参数，选择其它优化器则不用。

>学习率设置[参数对应代码](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/cvlibs/config.py)
```
lr_scheduler: # 学习率的相关设置
  type: PolynomialDecay # 一种学习率类型。共支持12种策略
  learning_rate: 0.01 # 初始学习率
  power: 0.9
  end_lr: 0
```
* 其中type表示学习率类型，可选PolynomialDecay等12种策略，如果type选用SGD需要添加momentum参数，选择其它优化器则不用。

>损失设置[参数对应代码](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/paddleseg/models/losses)
```
loss: 
  types:
    - type: CrossEntropyLoss  
  coef: [1, 1, 1]
```
* 其中types表示损失函数，可选CrossEntropyLoss、DiceLoss、FocalLoss等loss，其中coef表示用到的不同的损失函数对应的权重，所以 total_loss = coef_1 * loss_1 + .... + coef_n * loss_n

>模型说明设置[参数对应代码](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/paddleseg/cvlibs/config.py)
```
model:  
  type: PPLiteSeg  
  backbone: 
    type: STDC2
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz
```
* 其中type表示模型类别，有ann、attention_unet、bisenet、pp_liteseg、pspnet等多种模型供用户选择，其中backbone表示模型用到的骨干网络的名字和与训练权重文件。详细请参考[模型选择](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/paddleseg/models)[骨干网络选择](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/paddleseg/models/backbones)

**上面我们介绍的PP-LiteSeg配置文件，所有的配置信息都放置在同一个yml文件中。为了具有更好的复用性，PaddleSeg的配置文件采用了更加耦合的设计，配置文件支持包含复用。**

例如我们想要在cityscapes数据集上训练deeplabv3p模型，共涉及到两个配置文件，分别是：
>1.模型训练配置文件 PaddleSeg/configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml
```
_base_: '../_base_/cityscapes.yml'

batch_size: 2
iters: 80000

model:
  type: DeepLabV3P
  backbone:
    type: ResNet50_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz
  num_classes: 19
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]
  aspp_out_channels: 256
  align_corners: False
  pretrained: null
```
其中的```_base: ```后面是```deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml```配置文件会用到的数据和参数配置文件```cityscapes.yml```

> 2.数据和参数配置文件 cityscapes.yml
```
batch_size: 2
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
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
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
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
```
**注意：** 如果两个配置文件具有相同的字段信息，```_base_:```后面的配置文件信息会被覆盖。例如上述例子中的```deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml```配置文件会覆盖```cityscapes.yml```配置文件中的相同字段信息。

如下图所示：右侧```deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml```配置文件通过```_base_: '../_base_/cityscapes.yml'```来包含左侧```cityscapes.yml```配置文件，其中```_base_: ```设置的是被包含配置文件相对于该配置文件的路径。如果两个配置文件具有相同的字段信息，被包含的配置文件中的字段信息会被覆盖。如下图，1号配置文件可以覆盖2号配置文件的字段信息。
![配置文件图示](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/images/fig3.png)
