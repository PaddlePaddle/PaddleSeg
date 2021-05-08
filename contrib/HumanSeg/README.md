# HumanSeg人像分割

本教程基于PaddleSeg提供针对人像分割场景从预训练模型、Fine-tune的应用指南。最新发布HumanSeg-lite模型超轻量级人像分割模型，支持移动端场景的实时分割。

## 环境依赖

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/)。


#### 2. 安装PaddleSeg包

```shell
pip install paddleseg
```

#### 3. 下载PaddleSeg仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

## 预训练模型
HumanSeg开放了在大规模人像数据上训练的三个预训练模型，满足多种使用场景的需求

| 模型类型 | Checkpoint | Inference Model | Quant Inference Model | 备注 |
| --- | --- | --- | ---| --- |
| HumanSeg-server  | [humanseg_server_ckpt]() | [humanseg_server_inference]() | -- | 高精度模型，适用于服务端GPU且背景复杂的人像场景， 模型结构为Deeplabv3+/ResNet50, 输入大小（512， 512） |
| HumanSeg-mobile | [humanseg_mobile_ckpt]() | [humanseg_mobile_inference]() | [humanseg_mobile_quant]() | 轻量级模型, 适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_samll_v1，输入大小（192， 192）  |
| HumanSeg-lite | [humanseg_lite_ckpt]() | [humanseg_lite_inference]() |  [humanseg_lite_quant]() | 超轻量级模型, 适用于手机自拍人像，且有移动端实时分割场景， 模型结构为优化的ShuffleNetV2，输入大小（192， 192） |

| Method | Input Size | FLOPS | Parameters |
|-|-|-|-|
| ShuffleNetV2 | 398x224 | 293631872(294M) | 137012(137K) |
| ShuffleNetV2 | 192x192 | 121272192(121M) | 137012(137K) |
| HRNet w18 small v1 | 192x192 | 584182656(584M) | 1543954(1.54M) |

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

## 将模型导出为静态图模型

请确保位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python ../../export.py --config configs/shufflenetv2_humanseg_portrait.yml --save_dir export_model/shufflenetv2_humanseg_portrait --model_path saved_model/shufflenetv2_humanseg_portrait/best_model/model.pdparams --without_argmax --with_softmax
```

### 导出脚本参数解释

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|配置文件|是|-|
|save_dir|模型和visualdl日志文件的保存根路径|否|output|
|model_path|预训练模型参数的路径|否|配置文件中指定值|

## 结果文件

```shell
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```


## AIStudio在线教程

我们在AI Studio平台上提供了人像分割在线体验的教程，[点击体验]()
