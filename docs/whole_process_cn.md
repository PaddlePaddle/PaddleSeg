简体中文 | [English](whole_process.md)
# 20分钟快速上手PaddleSeg
PaddleSeg 通过模块组建，并通过配置化启动的方式，实现了从数据到模型部署的全部流程。本教程将以视盘分割为例，帮助大家轻松上手PaddleSeg。

本示例的主要流程如下：
1. 环境安装
2. 准备数据
3. 准备配置文件
4. 模型训练
5. 模型评估
6. 模型预测
7. 模型导出
8. 模型部署

如果大家想了解PaddleSeg API调用的使用方法，可以参考[教程](https://aistudio.baidu.com/aistudio/projectdetail/1339458?channelType=0&channel=0)。

## 1. 环境安装

参考[安装文档](./install_cn.md)进行环境配置。

## 2. 准备数据集
本示例将使用视盘分割（optic disc segmentation）数据集，它是我们目前支持的数十种数据集之一。实际使用过程中，大家可以参考[文档](./data/pre_data_cn.md)使用常见公开数据集，也可以参考[文档](./data/marker/marker_cn.md)准备自定义数据集。

该数据集是一组眼底医疗分割数据集，包含了267张训练图片、76张验证图片、38张测试图片。通过以下命令可以下载[视盘分割数据集](https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip)，解压保存到`PaddleSeg/data`目录下。

```
mkdir data
cd data
wget https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
unzip optic_disc_seg.zip
cd ..
```

数据集的原始图像和分割效果图如下所示，本示例的任务将是将眼球图片中的视盘区域分割出来。


<div align="center">
<img src="./images/fig1.png"  width = "400" />  
</div>


## 3. 准备配置文件

PaddleSeg提供**配置化驱动方式**进行模型训练、测试和预测等，配置文件是其中的关键。

我们先介绍本示例选用的模型，然后详细解读模型的配置文件。
### 3.1 PP-LiteSeg模型

本示例中，我们选择PP-LiteSeg模型进行训练。

PP-LiteSeg是PaddleSeg团队自研的轻量化模型，在Cityscapes数据集上超越其他模型，实现最优的精度和速度平衡。
具体而言，PP-LiteSeg模型沿用编解码架构，设计灵活的Decoder模块（FLD）、统一注意力融合模块（UAFM）和简易PPM上下文模块（SPPM），实现高精度和高效率的实时语义分割。

PP-LiteSeg模型的结构如下图。更多详细介绍，请参考[链接](../configs/pp_liteseg)。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148786-c8b91fd1-d006-4bad-8599-556daf959a75.png" width = "600" height = "300" alt="arch"  />
</div>

### 3.2 配置文件详细解读

PaddleSeg中，配置文件包括超参、训练数据集、验证数据集、优化器、损失函数、模型等信息。**所有模型在公开数据集上训练的配置文件在PaddleSeg/configs文件夹下面**。

大家可以灵活修改配置文件的内容，如自定义模型使用的骨干网络、模型使用的损失函数以及关于网络结构等配置，自定义配置数据处理的策略，如改变尺寸、归一化和翻转等数据增强的策略，这些修改可以参考对应模块的代码，传入相应参数即可。

以`PaddleSeg/configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml`为例，详细解读配置文件如下：
* yml文件定义了超参、训练数据集、测试数据集、优化器、学习率、损失函数、模型的配置信息，PaddleSeg基于配置信息构建对应的模块，进行模型训练、评估和导出。
* 超参主要包括batch_size和iters，前者是单卡的batch_size，后者表示训练迭代的轮数（单个batch进行一次前向和反向表示一轮）。
* 配置信息模块中，`type`字段对应到PaddleSeg代码中的类名（Python Class Name），其他字段对应类（Python Class）中`__init__`函数的初始化参数。
* 数据集dataset模块，支持的类在`PaddleSeg/paddleseg/datasets`目录下（使用@manager.DATASETS.add_component进行注册）。
* 数据预处理方式transforms模块，支持的类在`PaddleSeg/paddleseg/transforms/transforms.py`文件中（使用@manager.TRANSFORMS.add_component进行注册）。
* 优化器optimizer模块，支持Paddle提供的所有优化器类，具体参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#api
* 学习率衰减lr_scheduler模块，支持Paddle提供的所有lr_scheduler类，具体参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#about-lr
* 损失函数Loss模块，在`types`字段下分别定义使用的损失函数类，`coef`字段定义每个损失函数的权重。`types`字段下损失函数个数，应该等于`coef`字段数组的长度。如果所有损失函数相同，可以只定义一个损失函数。支持的损失函数类在`PaddleSeg/paddleseg/models/losses/`目录下（使用@manager.LOSSES.add_component注册）。
* 模型Model模块，支持的model类在`PaddleSeg/paddleseg/models/`目录下（使用@manager.MODELS.add_component注册），支持的backbone类在`PaddleSeg/paddleseg/models/backbones`目录下（使用@manager.BACKBONES.add_component注册）。

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

注意：
- 对于训练和测试数据集的预处理，PaddleSeg默认会添加读取图像操作、HWC转CHW的操作，所以这两个操作不用添加到transform配置字段中。
- 只有"PaddleSeg/configs/quick_start"下面配置文件中的学习率为单卡学习率，其他配置文件中均为4卡的学习率。如果大家单卡训练来复现公开数据集上的指标，学习率设置应变成原来的1/4。


上面我们介绍的PP-LiteSeg配置文件，所有的配置信息都放置在同一个yml文件中。为了具有更好的复用性，PaddleSeg的配置文件采用了更加耦合的设计，配置文件支持包含复用。

如下图，右侧`deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml`配置文件通过`_base_: '../_base_/cityscapes.yml'`来包含左侧`cityscapes.yml`配置文件，其中`_base_: `设置的是被包含配置文件相对于该配置文件的路径。

如果两个配置文件具有相同的字段信息，被包含的配置文件中的字段信息会被覆盖。如下图，1号配置文件可以覆盖2号配置文件的字段信息。

![](./images/fig3.png)


## 4. 模型训练

### 4.1 单卡训练

准备好配置文件后，在PaddleSeg根目录下执行如下命令，使用`tools/train.py`脚本进行单卡模型训练。

> 注意：PaddleSeg中模型训练、评估、预测、导出等命令，都要求在PaddleSeg根目录下执行。

```
export CUDA_VISIBLE_DEVICES=0 # Linux上设置1张可用的卡
# set CUDA_VISIBLE_DEVICES=0  # Windows上设置1张可用的卡

python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --save_interval 500 \
       --do_eval \
       --use_vdl \
       --save_dir output
```

上述训练命令解释：
* `--config`指定配置文件。
* `--save_interval`指定每训练特定轮数后，就进行一次模型保存或者评估（如果开启模型评估）。
* `--do_eval`开启模型评估。具体而言，在训练save_interval指定的轮数后，会进行模型评估。
* `--use_vdl`开启写入VisualDL日志信息，用于VisualDL可视化训练过程。
* `--save_dir`指定模型和visualdl日志文件的保存根路径。

在PP-LiteSeg示例中，训练的模型权重保存在output目录下，如下所示。总共训练1000轮，每500轮评估一次并保存模型信息，所以有`iter_500`和`iter_1000`文件夹。评估精度最高的模型权重，保存在`best_model`文件夹。后续模型的评估、测试和导出，都是使用保存在`best_model`文件夹下精度最高的模型权重。

```
output
  ├── iter_500          #表示在500步保存一次模型
    ├── model.pdparams  #模型参数
    └── model.pdopt     #训练阶段的优化器参数
  ├── iter_1000         #表示在1000步保存一次模型
    ├── model.pdparams  #模型参数
    └── model.pdopt     #训练阶段的优化器参数
  └── best_model        #精度最高的模型权重
    └── model.pdparams  
```

`train.py`脚本输入参数的详细说明如下。

| 参数名              | 用途                                                         | 是否必选项 | 默认值           |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| iters               | 训练迭代次数                                                 | 否         | 配置文件中指定值 |
| batch_size          | 单卡batch size                                               | 否         | 配置文件中指定值 |
| learning_rate       | 初始学习率                                                   | 否         | 配置文件中指定值 |
| config              | 配置文件                                                     | 是         | -                |
| save_dir            | 模型和visualdl日志文件的保存根路径                           | 否         | output           |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据   | 否         | 0                |
| use_vdl             | 是否开启visualdl记录训练数据                                 | 否         | 否               |
| save_interval       | 模型保存的间隔步数                                           | 否         | 1000             |
| do_eval             | 是否在保存模型时启动评估, 启动时将会根据mIoU保存最佳模型至best_model | 否         | 否               |
| log_iters           | 打印日志的间隔步数                                           | 否         | 10               |
| resume_model        | 恢复训练模型路径，如：`output/iter_1000`                     | 否         | None             |
| keep_checkpoint_max | 最新模型保存个数                                             | 否         | 5                |

### 4.2 多卡训练

使用多卡训练：首先通过环境变量`CUDA_VISIBLE_DEVICES`指定使用的多张显卡，如果不设置`CUDA_VISIBLE_DEVICES`，默认使用所有显卡进行训练；然后使用`paddle.distributed.launch`启动`train.py`脚本进行训练。

多卡训练的`tools/train.py`支持的输入参数和单卡训练相同。

由于Windows环境下不支持nccl，所以无法使用多卡训练。

举例如下，在PaddleSeg根目录下执行如下命令，进行多卡训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡
python -m paddle.distributed.launch tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### 4.3 恢复训练

如果训练中断，我们可以恢复训练，避免从头开始训练。

具体而言，通过给`tools/train.py`脚本设置`resume_model`输入参数，加载中断前最近一次保存的模型信息，恢复训练。

在PP-LiteSeg示例中，总共需要训练1000轮。假如训练到750轮中断了，我们在`output`目录下，可以看到在`iter_500`文件夹中保存了第500轮的训练信息。执行如下命令，加载第500轮的训练信息恢复训练。

单卡和多卡训练，都采用相同的方法设置`resume_model`输入参数，即可恢复训练。

```
python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --resume_model output/iter_500 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### 4.4 训练过程可视化

为了直观显示模型的训练过程，对训练过程进行分析从而快速的得到更好的模型，飞桨提供了可视化分析工具：VisualDL。

当`tools/train.py`脚本设置`use_vdl`输入参数后，PaddleSeg会将训练过程中的日志信息写入VisualDL文件，写入的日志信息包括：
* loss
* 学习率lr
* 训练时间
* 数据读取时间
* 验证集上mIoU（当打开了`do_eval`开关后生效）
* 验证集上mean Accuracy（当打开了`do_eval`开关后生效）

在PP-LiteSeg示例中，在训练过程中或者训练结束后，我们都可以通过VisualDL来查看日志信息。

首先执行如下命令，启动VisualDL；然后在浏览器输入提示的网址，效果如下图。

```
visualdl --logdir output/
```

![](./images/fig4.png)

## 5. 模型评估

训练完成后，大家可以使用评估脚本`tools/val.py`来评估模型的精度，即对配置文件中的验证数据集进行测试。

在PP-LiteSeg示例中，执行如下命令进行模型评估。其中，通过`--model_path`输入参数来指定评估的模型权重。

```
python tools/val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams
```

如果想使用多卡进行评估，可以使用`paddle.distributed.launch`启动`tools/val.py`脚本。

```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch tools/val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams
```

如果想进行多尺度翻转评估，可以通过传入`--aug_eval`进行开启，然后通过`--scales`传入尺度信息， `--flip_horizontal`开启水平翻转，`--flip_vertical`开启垂直翻转。使用示例如下：

```
python tools/val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams \
       --aug_eval \
       --scales 0.75 1.0 1.25 \
       --flip_horizontal
```

如果想进行滑窗评估，可以传入`--is_slide`进行开启， 通过`--crop_size`传入窗口大小， `--stride`传入步长。使用示例如下：

```
python tools/val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams \
       --is_slide \
       --crop_size 256 256 \
       --stride 128 128
```

在图像分割领域中，评估模型质量主要是通过三个指标进行判断，`准确率`（acc）、`平均交并比`（Mean Intersection over Union，简称mIoU）、`Kappa系数`。

- **准确率**：指类别预测正确的像素占总像素的比例，准确率越高模型质量越好。
- **平均交并比**：对每个类别数据集单独进行推理计算，计算出的预测区域和实际区域交集除以预测区域和实际区域的并集，然后将所有类别得到的结果取平均。在本例中，正常情况下模型在验证集上的mIoU指标值会达到0.80以上，显示信息示例如下所示，第3行的**mIoU=0.9232**即为mIoU。
- **Kappa系数**：一个用于一致性检验的指标，可以用于衡量分类的效果。kappa系数的计算是基于混淆矩阵的，取值为-1到1之间，通常大于0。其公式如下所示，P0P_0*P*0为分类器的准确率，PeP_e*P**e*为随机分类器的准确率。Kappa系数越高模型质量越好。

<a href="https://www.codecogs.com/eqnedit.php?latex=Kappa=&space;\frac{P_0-P_e}{1-P_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Kappa=&space;\frac{P_0-P_e}{1-P_e}" title="Kappa= \frac{P_0-P_e}{1-P_e}" /></a>

随着评估脚本的运行，最终打印的评估日志如下。

```
...
2022-06-22 11:05:55 [INFO]      [EVAL] #Images: 76 mIoU: 0.9232 Acc: 0.9970 Kappa: 0.9171 Dice: 0.9585
2022-06-22 11:05:55 [INFO]      [EVAL] Class IoU:
[0.997  0.8494]
2022-06-22 11:05:55 [INFO]      [EVAL] Class Precision:
[0.9984 0.9237]
2022-06-22 11:05:55 [INFO]      [EVAL] Class Recall:
[0.9986 0.9135]
```

## 6. 模型预测

除了分析模型的IOU、ACC和Kappa指标之外，我们还可以可视化一些具体样本的分割效果，从Bad Case启发进一步优化的思路。

`predict.py`脚本是专门用来可视化预测的，命令格式如下所示。

```
python tools/predict.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams \
       --image_path data/optic_disc_seg/JPEGImages/H0002.jpg \
       --save_dir output/result
```

其中`image_path`可以是一个图片路径，也可以是一个目录。如果是一个目录，将对目录内的所有图片进行预测并保存可视化结果图。

同样的，可以通过`--aug_pred`开启多尺度翻转预测， `--is_slide`开启滑窗预测。

我们选择1张图片进行查看，如下图。

<div align="center">
<img src="./images/fig5.png"  width = "600" />  
</div>

## 7 模型导出

上述模型训练、评估和预测，都是使用飞桨的动态图模式。动态图模式具有灵活、方便的优点，但是不适合工业级部署的速度要求。

为了满足工业级部署的需求，飞桨提供了动转静的功能，即将训练出来的动态图模型转化成静态图预测模型。预测引擎加载、执行预测模型，实现更快的预测速度。

执行如下命令，加载精度最高的模型权重，导出预测模型。
```
python tools/export.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/best_model/model.pdparams \
       --save_dir output/infer_model
```

参数说明如下：
| 参数名     | 用途                               | 是否必选项 | 默认值           |
| :--------- | :--------------------------------- | :--------- | :--------------- |
| config     | 配置文件                           | 是         | -                |
| model_path | 预训练模型权重的路径，使用精度最高的模型权重  | 否         | 配置文件中指定值 |
| save_dir   | 保存预测模型的路径 | 否         | output           |


```
output/infer_model
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件，可以使用netron软件进行可视化查看
```

## 8 模型部署

PaddleSeg目前支持以下部署方式：
| 端侧         | 库           | 教程   |
| :----------- | :----------- | :----- |
| Python端部署 | Paddle预测库 | [示例](../deploy/python/) |
| C++端部署 | Paddle预测库 | [示例](../deploy/cpp/) |
| 移动端部署   | PaddleLite   | [示例](../deploy/lite/) |
| 服务端部署   | HubServing   | 完善中 |
| 前端部署     | PaddleJS     | [示例](../deploy/web/) |

比如使用Python端部署方式，运行如下命令，会在output文件下面生成一张H0002.png的分割图像。

```
python deploy/python/infer.py \
  --config output/infer_model/deploy.yaml \
  --image_path data/optic_disc_seg/JPEGImages/H0002.jpg \
  --save_dir output/result
```

## 9 二次开发

在尝试完成使用配置文件进行训练之后，肯定有小伙伴想基于PaddleSeg进行更深入的开发，在这里，我们大概介绍一下PaddleSeg代码结构，

```
PaddleSeg
     ├──  configs #配置文件文件夹
     ├──  paddleseg #训练部署的核心代码
        ├── core # 启动模型训练，评估与预测的接口
        ├── cvlibs # Config类定义在该文件夹中。它保存了数据集、模型配置、主干网络、损失函数等所有的超参数。
            ├── callbacks.py
            └── ...
        ├── datasets #PaddleSeg支持的数据格式，包括ade、citycapes等多种格式
            ├── ade.py
            ├── citycapes.py
            └── ...
        ├── models #该文件夹下包含了PaddleSeg组网的各个部分
            ├── backbone # paddleseg的使用的主干网络
            ├── hrnet.py
            ├── resnet_vd.py
            └── ...
            ├── layers # 一些组件，例如attention机制
            ├── activation.py
            ├── attention.py
            └── ...
            ├── losses #该文件夹下包含了PaddleSeg所用到的损失函数
            ├── dice_loss.py
            ├── lovasz_loss.py
            └── ...
            ├── ann.py #该文件表示的是PaddleSeg所支持的算法模型，这里表示ann算法。
            ├── deeplab.py #该文件表示的是PaddleSeg所支持的算法模型，这里表示Deeplab算法。
            ├── unet.py #该文件表示的是PaddleSeg所支持的算法模型，这里表示unet算法。
            └── ...
        ├── transforms #进行数据预处理的操作，包括各种数据增强策略
            ├── functional.py
            └── transforms.py
        └── utils
            ├── visualize.py
            └── ...
     ├──  tools
          ├──  train.py  # 训练入口文件，该文件里描述了参数的解析，训练的启动方法，以及为训练准备的资源等。
          ├──  predict.py # 预测文件
     └── ...
```

同学们还可以尝试使用PaddleSeg的API来自己开发，开发人员在使用pip install命令安装PaddleSeg后，仅需通过几行代码即可轻松实现图像分割模型的训练、评估和推理。 感兴趣的小伙伴们可以访问[PaddleSeg动态图API使用教程](https://aistudio.baidu.com/aistudio/projectdetail/1339458?channelType=0&channel=0)

PaddleSeg等各领域的开发套件已经为真正的工业实践提供了顶级方案，有国内的团队使用PaddleSeg的开发套件取得国际比赛的好成绩，可见开发套件提供的效果是State Of The Art的。
