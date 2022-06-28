> 本目录下的配置文件用于飞桨模型选型工具（PaddleSMRT）。
# PaddleSMRT

## 一、项目介绍

[PaddleSMRT](https://www.paddlepaddle.org.cn/smrt) 是飞桨结合产业落地经验推出的产业模型选型工具，在项目落地过程中，用户根据自身的实际情况，输入自己的需求，即可以得到对应在算法模型、部署硬件以及教程文档的信息。
同时为了更加精准的推荐，增加了数据分析功能，用户上传自己的标注文件，系统可以自动分析数据特点，例如数据分布不均衡、小目标、密集型等，从而提供更加精准的模型以及优化策略，更好的符合场景的需求。


本文档主要介绍PaddleSMRT在分割方向上是如何进行模型选型推荐，以及推荐模型的使用方法。

## 二、数据介绍

PaddleSMRT结合产业真实场景，通过比较算法效果，向用户推荐最适合的模型。目前PaddleSMRT覆盖工业质检、城市安防两大场景，下面介绍PaddleSMRT进行算法对比所使用的数据集。

### 1. 新能源电池质检数据集

数据集为新能源电池电池组件质检数据集，包含15021张图片，覆盖45种缺陷类型，例如掉胶，裂纹，划痕等。

新能源电池数据展示图:

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169200335-c77d58e4-8916-46e4-be4b-eb7fe48a2f80.png"  width = "600" />  
</div>

数据集特点为：

1. 类别分布均衡
2. 属于小目标数据
3. 非密集型数据

### 2. 铝件质检数据集

数据集为铝件生产过程中的质检数据集，包含11293张图片，覆盖5种缺陷类型，例如划伤，压伤，起皮等。

铝件质检数据展示图:

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169200252-95a69964-0ae1-40bb-b2b9-2ba17f9bef64.png"  width = "600" />  
</div>


数据集特点为：

1. 类别分布不均衡
2. 属于小目标数据
3. 非密集型数据


## 三、推荐模型使用全流程

通过飞桨官网的[模型选型工具](https://www.paddlepaddle.org.cn/smrt)明确需要使用的模型后，大家需要进行数据准备、模型训练、模型导出、模型部署，下面我们以一个例子进行简要说明。

### 3.1 准备环境

参考PaddleSeg[安装文档](../../docs/install_cn.md)安装PaddlePaddle、下载PaddleSeg代码、安装PaddleSeg依赖库。

### 3.2 准备数据

详细的数据准备方法，请参考[数据标注文档](../../docs/data/marker/marker_cn.md)和[数据配置文档](../../docs/data/custom/data_prepare_cn.md)。

此处，我们准备了一个缺陷分割的数据集，点击[链接](https://paddle-smrt.bj.bcebos.com/data/seg/defect_data.zip)下载，或者执行如下命令下载。
```
wget https://paddle-smrt.bj.bcebos.com/data/seg/defect_data.zip
```

将下载的数据集解压，并存放到`PaddleSeg/data/defect_data`目录，如下。
```
PaddleSeg/data/defect_data
├── Annotations
├── JPEGImages
├── test.txt
├── train.txt
└── val.txt
```

该数据集的原始图片保存在JPEGImages文件，标注图片保存在Annotations。

train.txt、val.txt和test.txt分别标识用于训练、验证和测试的数据，其中内容如下，每一行的前面表示原始图像的相对路径、后面表示标注图像的相对路径。
```
JPEGImages/liewen_26.png Annotations/liewen_26.png
JPEGImages/diaojiao_394.png Annotations/diaojiao_394.png
```

标注图像包含背景和3类缺陷目标，总共4类，分别标注的数值使0，1，2，3。

### 3.3 准备配置文件

PaddleSeg推荐使用配置文件的方式来训练、导出模型，简单方便。

针对工业质检任务，我们为模型选型工具推荐的6个模型准备好了配置文件，存放在`PaddleSeg/configs/smrt`目录下。

```
PaddleSeg/configs/smrt
├── base_cfg.yml  
├── pp_liteseg_stdc1.yml  
├── pp_liteseg_stdc2.yml  
├── deeplabv3p_resnet50_os8.yml  
├── ocrnet_hrnetw18.yml
├── bisenetv2.yml
└── sfnet_resnet18_os8.yml
```

其中，`base_cfg.yml`是公共配置，包括数据集、优化器、学习率，其他文件包含`base_cfg.yml`，额外定义了模型、损失函数。

在其他应用中，大家可以根据实际情况修改上述配置文件中的字段，而且需要根据模型数据量调整配置文件中的超参，比如训练轮数iters、batch_size、学习率等。

### 3.4 执行训练

本教程简单演示单卡和多卡训练，详细的模型训练方法请参考[文档](../../docs/train/train_cn.md)。

**单卡训练**

在PaddleSeg目录下，执行如下命令，使用单卡GPU进行模型训练。

```
export CUDA_VISIBLE_DEVICES=0 # Linux下设置1张可用的卡
# set CUDA_VISIBLE_DEVICES=0  # windows下设置1张可用的卡

cd PaddleSeg

python train.py \
       --config configs/smrt/pp_liteseg_stdc2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 1000 \
       --save_dir output/pp_liteseg_stdc2
```

说明：
* 上面脚本选择`pp_liteseg_stdc2`模型进行训练，所以加载`configs/smrt/pp_liteseg_stdc2.yml`配置文件，如果需要使用其他模型，可以修改`--config`输入参数。
* `--do_eval`表示训练过程中会进行测试，`--save_interval`设置每训练多少轮会进行一次测试。
* 训练结束后，精度最高的模型保存在`--save_dir`中，比如`output/pp_liteseg_stdc2/best_model/`。
* 查看训练的log，可以知道模型训练的最高精度。
* 为了实现最高的精度，大家可以适当调参，重点关注学习率lr、BatchSize、训练轮数iters、损失函数Loss等。


**多卡训练**

在PaddleSeg目录下，执行如下命令，使用多卡GPU进行模型训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡

python -m paddle.distributed.launch train.py \
       --config configs/smrt/pp_liteseg_stdc2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 1000 \
       --save_dir output/pp_liteseg_stdc2
```

### 3.5 模型导出

训练得到精度符合预期的模型后，可以导出预测模型，进行部署。详细的模型导出方法请参考[文档](../../docs/model_export_cn.md)。

```
python export.py \
       --config configs/smrt/pp_liteseg_stdc2.yml \
       --model_path output/pp_liteseg_stdc2/best_model/model.pdparams \
       --save_dir output/pp_liteseg_stdc2/infer_models
```

上面脚本加载`pp_liteseg_stdc2`模型精度最高的权重，导出预测模型保存在`output/pp_liteseg_stdc2/infer_models`目录。

### 3.5 部署

导出模型后，大家可以参考如下文档进行部署。

| 端侧         | 库           | 教程   |
| :----------- | :----------- | :----- |
| 服务端端Python部署 | Paddle预测库 | [文档](../../docs/deployment/inference/python_inference_cn.md) |
| 服务器端端C++部署 | PaddleInference预测库 | [文档](../../docs/deployment/inference/cpp_inference_cn.md) |
| 移动端部署   | PaddleLite   | [文档](../../docs/deployment/lite/lite_cn.md) |
| 前端部署     | PaddleJS     | [文档](../../docs/deployment/web/web_cn.md) |

## 四、部署demo

为了更方便大家部署，我们也提供了完备的可视化部署Demo，欢迎尝试使用

* [Windows Demo下载地址](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp/docs/csharp_deploy)

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169064583-c931f4c0-dfd6-4bfa-85f1-be68eb351e4a.png"  width = "800" />  
</div>

* [Linux Demo下载地址](https://github.com/cjh3020889729/The-PaddleX-QT-Visualize-GUI)

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169065951-147f8d51-bf3e-4a28-9197-d717968de73f.png"  width = "800" />  
</div>

## 五、场景范例

为了更方便大家更好的进行产业落地，PaddleSMRT也提供了详细详细的应用范例，欢迎大家使用。

* 工业视觉
  * [工业缺陷检测](https://aistudio.baidu.com/aistudio/projectdetail/2598319)
  * [表计读数](https://aistudio.baidu.com/aistudio/projectdetail/2598327)
  * [钢筋计数](https://aistudio.baidu.com/aistudio/projectdetail/2404188)
