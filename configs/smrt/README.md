> 本目录下的配置文件用于飞桨模型选型工具（Paddle-SMRT）。
# 1 简介

模型选型工具是根据任务的类别、部署环境等信息，给大家推荐合适的模型，可以避免大家训练测试多个模型的麻烦。

通过模型选型工具，大家选择好目标模型后，接下来需要进行模型训练和部署。

本文档在工业质检场景，针对选型工具推荐的分割模型，用一个示例来简单介绍模型的训练、导出和部署。

# 2 准备
## 准备环境

参考PaddleSeg[安装文档](../../docs/install_cn.md)安装PaddlePaddle、下载PaddleSeg代码、安装PaddleSeg依赖库。

## 准备数据

我们准备了一个缺陷分割的数据集，点击[链接](https://paddle-smrt.bj.bcebos.com/data/seg/defect_data.zip)下载，或者执行如下命令下载。
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

详细的数据准备方法，请参考[数据标注文档](../../docs/data/marker/marker_cn.md)和[数据配置文档](../../docs/data/custom/data_prepare_cn.md)。

# 3 模型训练

## 配置文件

PaddleSeg可以使用配置文件的方式来训练模型，简单方便。

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

在实际应用中，大家需要根据模型数据量调整配置文件中的超参，比如训练轮数iters。

## 执行训练

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

# 4 模型导出

训练得到精度符合预期的模型后，可以导出预测模型，进行部署。详细的模型导出方法请参考[文档](../../docs/model_export_cn.md)。

```
python export.py \
       --config configs/smrt/pp_liteseg_stdc2.yml \
       --model_path output/pp_liteseg_stdc2/best_model/model.pdparams \
       --save_dir output/pp_liteseg_stdc2/infer_models
```

上面脚本加载`pp_liteseg_stdc2`模型精度最高的权重，导出预测模型保存在`output/pp_liteseg_stdc2/infer_models`目录。


# 5 部署demo

导出模型后，大家可以参考如下文档进行部署。

| 端侧         | 库           | 教程   |
| :----------- | :----------- | :----- |
| 服务端端Python部署 | Paddle预测库 | [文档](../../docs/deployment/inference/python_inference_cn.md) |
| 服务器端端C++部署 | PaddleInference预测库 | [文档](../../docs/deployment/inference/cpp_inference_cn.md) |
| 移动端部署   | PaddleLite   | [文档](../../docs/deployment/lite/lite_cn.md) |
| 前端部署     | PaddleJS     | [文档](../../docs/deployment/web/web_cn.md) |

为了更方便大家部署，我们也提供了可视化部署Demo，欢迎尝试使用。

* [Windows Demo下载地址](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp/docs/csharp_deploy)

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169064583-c931f4c0-dfd6-4bfa-85f1-be68eb351e4a.png"  width = "800" />  
</div>

* [Linux Demo下载地址](https://github.com/cjh3020889729/The-PaddleX-QT-Visualize-GUI)

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169065951-147f8d51-bf3e-4a28-9197-d717968de73f.png"  width = "800" />  
</div>
