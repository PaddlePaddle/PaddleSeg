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
| --- | --- | --- | ---| --- |
| HumanSeg-server  | [humanseg_server_ckpt](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_server_ckpt.zip) | [humanseg_server_inference](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_server_inference.zip) | -- | 高精度模型，适用于服务端GPU且背景复杂的人像场景， 模型结构为Deeplabv3+/Xcetion65, 输入大小（512， 512） |
| HumanSeg-mobile | [humanseg_mobile_ckpt](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_ckpt.zip) | [humanseg_mobile_inference](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_inference.zip) | [humanseg_mobile_quant](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_quant.zip) | 轻量级模型, 适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_samll_v1，输入大小（192， 192）  |
| HumanSeg-lite | [humanseg_lite_ckpt](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite_ckpt.zip) | [humanseg_lite_inference](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite_inference.zip) |  [humanseg_lite_quant](https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_lite_quant.zip) | 超轻量级模型, 适用于手机自拍人像，且有移动端实时分割场景， 模型结构为优化的ShuffleNetV2，输入大小（192， 192） |


模型性能

| 模型 | 模型大小 | 计算耗时 |
| --- | --- | --- |
|humanseg_server_inference| 158M | - |
|humanseg_mobile_inference | 5.8 M | 42.35ms |
|humanseg_mobile_quant | 1.6M | 24.93ms |
|humanseg_lite_inference | 541K | 17.26ms |
|humanseg_lite_quant | 187k | 11.89ms |

计算耗时运行环境： 小米，cpu：骁龙855， 内存：6GB， 图片大小：192*192)


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
结合DIS（Dense Inverse Search-basedmethod）光流算法预测结果与分割结果，改善视频流人像分割
```bash
# 通过电脑摄像头进行实时分割处理
python video_infer.py --model_dir pretrained_weights/humanseg_lite_inference

# 对人像视频进行分割处理
python video_infer.py --model_dir pretrained_weights/humanseg_lite_inference --video_path data/video_test.mp4
```

视频分割结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/result.gif" width="20%" height="20%">

根据所选背景进行背景替换，背景可以是一张图片，也可以是一段视频。
```bash
# 通过电脑摄像头进行实时背景替换处理, 也可通过'--background_video_path'传入背景视频
python bg_replace.py --model_dir pretrained_weights/humanseg_lite_inference --background_image_path data/background.jpg

# 对人像视频进行背景替换处理, 也可通过'--background_video_path'传入背景视频
python bg_replace.py --model_dir pretrained_weights/humanseg_lite_inference --video_path data/video_test.mp4 --background_image_path data/background.jpg

# 对单张图像进行背景替换
python bg_replace.py --model_dir pretrained_weights/humanseg_lite_inference --image_path data/human_image.jpg --background_image_path data/background.jpg

```

背景替换结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/bg_replace.gif" width="20%" height="20%">


**NOTE**:

视频分割处理时间需要几分钟，请耐心等待。

提供的模型适用于手机摄像头竖屏拍摄场景，宽屏效果会略差一些。

## 训练
使用下述命令基于与训练模型进行Fine-tuning，请确保选用的模型结构`model_type`与模型参数`pretrained_weights`匹配。
```bash
python train.py --model_type HumanSegMobile \
--save_dir output/ \
--data_dir data/mini_supervisely \
--train_list data/mini_supervisely/train.txt \
--val_list data/mini_supervisely/val.txt \
--pretrained_weights pretrained_weights/humanseg_mobile_ckpt \
--batch_size 8 \
--learning_rate 0.001 \
--num_epochs 10 \
--image_shape 192 192
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
* `--image_shape`: 网络输入图像大小（w, h）

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
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--val_list`: 验证集列表路径
* `--image_shape`: 网络输入图像大小（w, h）

## 预测
使用下述命令进行预测， 预测结果默认保存在`./output/result/`文件夹中。
```bash
python infer.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--test_list data/mini_supervisely/test.txt \
--save_dir output/result \
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--data_dir`: 数据集路径
* `--test_list`: 测试集列表路径
* `--image_shape`: 网络输入图像大小（w, h）

## 模型导出
```bash
python export.py --model_dir output/best_model \
--save_dir output/export
```
其中参数含义如下：
* `--model_dir`: 模型路径
* `--save_dir`: 导出模型保存路径

## 离线量化
```bash
python quant_offline.py --model_dir output/best_model \
--data_dir data/mini_supervisely \
--quant_list data/mini_supervisely/val.txt \
--save_dir output/quant_offline \
--image_shape 192 192
```
其中参数含义如下：
* `--model_dir`: 待量化模型路径
* `--data_dir`: 数据集路径
* `--quant_list`: 量化数据集列表路径，一般直接选择训练集或验证集
* `--save_dir`: 量化模型保存路径
* `--image_shape`: 网络输入图像大小（w, h）

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
--image_shape 192 192
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
* `--image_shape`: 网络输入图像大小（w, h）

## AIStudio在线教程

我们在AI Studio平台上提供了人像分割在线体验的教程，[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/475345)
