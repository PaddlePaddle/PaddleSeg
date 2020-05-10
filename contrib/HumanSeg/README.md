# HumanSeg

本教程旨在通过paddlepaddle框架实现人像分割从训练到部署的流程。

HumanSeg从复杂到简单提供三种人像分割模型：HumanSegServer、HumanSegMobile、HumanSegLite,
HumanSegServer适应于服务端，HumanSegMobile和HumanSegLite适用于移动端。

## 环境依赖

* PaddlePaddle >= 1.7.0 或develop版本
* Python 3.5+

通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令
```shell
$ pip install -r requirements.txt
```

## 模型
| 模型类型 | 预训练模型 | 导出模型 | 量化模型 | 说明 |
| --- | --- | --- | --- | --- |
| HumanSegServer | [humanseg_server]() | [humanseg_server_export]() | [humanseg_server_quant]() | 服务端GPU环境  |
| HumanSegMobile | [humanseg_mobile]() | [humanseg_mobile_export]() | [humanseg_mobile_quant]() | 小模型, 适合轻量级计算环境 |
| HumanSegLite | [humanseg_lite]() | [humanseg_lite_export]() |  [humanseg_lite_quant]() | 小模型, 适合轻量级计算环境 |

## 预测部署
人像分割预测部署请参见[deploy](./deploy)

## 准备训练数据
我们提供了一份demo数据集，通过运行以下代码进行下载，该数据集是从supervise.ly抽取的一个小数据集。

```bash
python data/download_data.py
```

## 下载预训练模型
运行以下代码进行预训练模型的下载
```bash
python pretrained_weights/download_pretrained_weights.py
```

## 训练
使用下述命令进行训练
```bash
CUDA_VISIBLE_DEVICES=0 && python train.py --model_type HumanSegMobile \
--save_dir output/ \
--data_dir data/mini_supervisely \
--train_list data/mini_supervisely/train.txt \
--val_list data/mini_supervisely/val.txt \
--pretrained_weights pretrained_weights/humanseg_Mobile \
--batch_size 8 \
--learning_rate 0.001 \
--num_epochs 10 \
--save_interval_epochs 2
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
* `--save_interval_epochs`: 模型保存间隔

更多参数请运行下述命令进行参看：
```bash
python train.py --help
```

## 评估
使用下述命令进行评估
```bash
python val.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--val_list data/mini_supervisely/val.txt \
--batch_size 2
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--val_list`: 验证集列表路径
* `--batch_size`: 批大小

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
--save_interval_epochs 1
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
* `--save_interval_epochs`: 模型保存间隔
