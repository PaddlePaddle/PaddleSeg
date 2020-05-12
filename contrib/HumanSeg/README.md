# HumanSeg人像分割模型

本教程基于PaddleSeg核心分割网络，提供针对人像分割场景从预训练模型、Fine-tune、视频分割预测部署的全流程应用指南。最新发布HumanSeg-lite模型超轻量级人像分割模型，支持移动端场景的实时分割。

## 环境依赖

* Python == 3.5/3.6/3.7
* PaddlePaddle >= 1.7.2

PaddlePaddle的安装可参考[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick)

通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令
```shell
$ pip install -r requirements.txt
```

## 预训练模型
HumanSeg开放了在大规模人像数据上训练的三个预训练模型，满足多种使用场景的需求
| 模型类型 | Checkpoint | Inference Model | Quant Inference Model | 备注 |
| --- | --- | --- | --- | --- |
| HumanSeg-server | [humanseg_server_ckpt](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_server.zip) | [humanseg_server_export](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_server_export.zip) | -- | 高精度模型，适用于服务端GPU且背景复杂的人像场景  |
| HumanSeg-mobile | [humanseg_mobile_ckpt](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile.zip) | [humanseg_mobile_export](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_export.zip) | [humanseg_mobile_quant](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_quant.zip) | 轻量级模型, 适用于移动端或服务端CPU的前置摄像头场景 |
| HumanSeg-lite | [humanseg_lite](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite.zip) | [humanseg_lite_export](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite_export.zip) |  [humanseg_lite_quant](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite_quant.zip) | 超轻量级模型, 适用于手机自拍人像，且有移动端实时分割场景 |

**NOTE:**
其中Checkpoint为模型权重，用于Fine-tuning场景。

* Inference Model和Quant Inference Model为预测部署模型，包含`__model__`计算图结构、`__params__`模型参数和`model.yaml`基础的模型配置信息。

* 其中Inference Model适用于服务端的CPU和GPU预测部署，Qunat Inference Model为量化版本，适用于通过Paddle Lite进行移动端等端侧设备部署。更多Paddle Lite部署说明查看[Paddle Lite文档](https://paddle-lite.readthedocs.io/zh/latest/)

执行以下脚本进行HumanSeg预训练模型的下载
```bash
python pretrained_weights/download_pretrained_weights.py
```

## 下载测试数据
我们提供了[supervise.ly](https://supervise.ly/)发布人像分割数据集**Supervisely Persons**, 从中随机抽取一小部分并转化成PaddleSeg可直接加载数据格式。通过运行以下代码进行快速下载，其中包含手机前置摄像头的人像测试视频`video_test.mp4`.

```bash
python data/download_data.py
```

## 快速体验视频流人像分割
```bash
# 通过电脑摄像头进行实时分割处理
python video_infer.py --model_dir pretrained_weights/humanseg_lite_epxort/

# 对人像视频进行分割处理
python video_infer.py --model_dir pretrained_weights/humanseg_lite_epxort/ --video_path data/video_test.mp4
```

## 训练
使用下述命令基于与训练模型进行Fine-tuning，请确保选用的模型结构`model_type`与模型参数`pretrained_weights`匹配。
```bash
python train.py --model_type HumanSegMobile \
--save_dir output/ \
--data_dir data/mini_supervisely \
--train_list data/mini_supervisely/train.txt \
--val_list data/mini_supervisely/val.txt \
--pretrained_weights pretrained_weights/humanseg_Mobile \
--batch_size 8 \
--learning_rate 0.001 \
--num_epochs 10 \
```
其中参数含义如下：
* `--model_type`: 模型类型，可选项为：HumanSegServer、HumanSegMobile和HumanSegLite
* `--save_dir`: 模型保存路径
* `--data_dir`: 数据集路径
* `--train_list`: 训练集列表路径
* `--val_list`: 验证集列表路径
* `--pretrained_weights`: 预训练模型路径
* `--batch_size`: 批大小
* `--learning_rate`: 初始学习率
* `--num_epochs`: 训练轮数

更多命令行帮助可运行下述命令进行查看：
```bash
python train.py --help
```
**NOTE**
可通过更换`--model_type`变量与对应的`--pretrained_weights`使用不同的模型快速尝试。

## 评估
使用下述命令进行评估
```bash
python val.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--val_list data/mini_supervisely/val.txt \
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--val_list`: 验证集列表路径

## 预测
使用下述命令进行预测
```bash
python infer.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--test_list data/mini_supervisely/test.txt
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--test_list`: 测试集列表路径

## 模型导出
```bash
python export.py --model_dir output/best_model \
--save_dir output/export
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--save_dir`: 导出模型保存路径

## 离线量化
```bash
python quant_offline.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--quant_list data/mini_supervisely/val.txt \
--save_dir output/quant_offline
```
其中参数含义如下：
* `--model_dir`: 待量化模型路径
* `--data_dir`: 数据集路径
* `--quant_list`: 量化数据集列表路径，一般直接选择训练集或验证集
* `--save_dir`: 量化模型保存路径

## 在线量化
利用float训练模型进行在线量化。
```bash
python quant_online.py --model_type HumanSegMobile \
--save_dir output/quant_online \
--data_dir data/mini_supervisely \
--train_list data/mini_supervisely/train.txt \
--val_list data/mini_supervisely/val.txt \
--pretrained_weights output/best_model \
--batch_size 2 \
--learning_rate 0.001 \
--num_epochs 2 \
```
其中参数含义如下：
* `--model_type`: 模型类型，可选项为：HumanSegServer、HumanSegMobile和HumanSegLite
* `--save_dir`: 模型保存路径
* `--data_dir`: 数据集路径
* `--train_list`: 训练集列表路径
* `--val_list`: 验证集列表路径
* `--pretrained_weights`: 预训练模型路径,
* `--batch_size`: 批大小
* `--learning_rate`: 初始学习率
* `--num_epochs`: 训练轮数
