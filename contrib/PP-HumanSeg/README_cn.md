简体中文 | [English](README.md)

# 人像分割PP-HumanSeg

人像分割是图像分割领域的高频应用，PaddleSeg推出在大规模人像数据上训练的人像分割系列模型PP-HumanSeg，包括超轻量级模型**PP-HumanSeg-Lite**，满足在服务端、移动端、Web端多种使用场景的需求。我们提供从训练到部署的全流程应用指南，以及视频流人像分割、背景替换教程。可基于Paddle.js在网页体验[人像扣图](https://paddlejs.baidu.com/humanseg)效果、[视频背景替换及弹幕穿透](https://www.paddlepaddle.org.cn/paddlejs)效果。

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/149886667-f47cab88-e81a-4fd7-9f32-fbb34a5ed7ce.png"  height="300">        <img src="https://user-images.githubusercontent.com/30695251/149887482-d1fcd5d3-2cce-41b5-819b-bfc7126b7db4.png"  height="300">
</p>

新冠疫情催化远程办公需求，视频会议产品迅速爆发。百度视频会议可实现Web端一秒入会，其中的虚拟背景功能采用我们的PP-HumanSeg-Lite模型，实现实时背景替换和背景虚化功能，保护用户隐私，以及增加会议中的趣味性。

<p align="center">
<img src="https://github.com/LutaoChu/transfer_station/raw/master/conference.gif" width="60%" height="60%">
</p>

## 最新动向
- [2022-1-4] 人像分割论文[PP-HumanSeg](./paper.md)发表于WACV 2022 Workshop，并开源连通性学习（SCL）方法和大规模视频会议数据集。

## 目录
- [人像分割模型](#人像分割模型)
  - [通用人像分割](#通用人像分割)
  - [肖像分割](#肖像分割)
- [安装](#安装)
- [快速体验](#快速体验)
  - [视频流人像分割](#视频流人像分割)
  - [视频流背景替换](#视频流背景替换)
  - [在线运行教程](#在线运行教程)
- [训练评估预测演示](#训练评估预测演示)
- [模型导出](#模型导出)
- [Web端部署](#Web端部署)
- [移动端部署](#移动端部署)

## 人像分割模型
### 通用人像分割
针对通用人像分割任务，PP-HumanSeg开放了在大规模人像数据上训练的三个人像模型，满足服务端、移动端、Web端多种使用场景的需求。

| 模型名 | 模型说明 | Checkpoint | Inference Model |
| --- | --- | --- | ---|
| PP-HumanSeg-Server | 高精度模型，适用于服务端GPU且背景复杂的场景， 模型结构为Deeplabv3+/ResNet50, 输入大小（512， 512） |[server_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip) | [server_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.zip) |
| PP-HumanSeg-Mobile | 轻量级模型，适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_samll_v1，输入大小（192， 192）  | [mobile_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip) | [mobile_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/fcn_hrnetw18_small_v1_humanseg_192x192_with_softmax.zip) |
| PP-HumanSeg-Lite | 超轻量级模型，适用于Web端或移动端实时分割场景，例如手机自拍、Web视频会议，模型结构为[Paddle自研模型](../../configs/pp_humanseg_lite/README.md)，输入大小（192， 192） | [lite_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/pphumanseg_lite_generic_192x192.zip) | [lite_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/pphumanseg_lite_generic_192x192_with_softmax.zip) |


NOTE:
* 其中Checkpoint为模型权重，用于Fine-tuning场景。

* Inference Model为预测部署模型，包含`model.pdmodel`计算图结构、`model.pdiparams`模型参数和`deploy.yaml`基础的模型配置信息。

* 其中Inference Model适用于服务端的CPU和GPU预测部署，适用于通过Paddle Lite进行移动端等端侧设备部署。更多Paddle Lite部署说明查看[Paddle Lite文档](https://paddle-lite.readthedocs.io/zh/latest/)

#### 模型性能

| 模型名 |Input Size | FLOPs | Parameters | 计算耗时 | 模型大小 |
|-|-|-|-|-|-|
| PP-HumanSeg-Server | 512x512 | 114G | 26.8M | 37.96ms | 103Mb |
| PP-HumanSeg-Mobile | 192x192 | 584M | 1.54M | 13.17ms | 5.9Mb |
| PP-HumanSeg-Lite | 192x192 | 121M | 137K | 10.51ms | 543Kb |

测试环境：Nvidia Tesla V100单卡。

### 肖像分割
针对肖像分割(Portrait Segmentation)任务，PP-HumanSeg开放了肖像分割模型，该模型已应用于百度视频会议。

| 模型名 | 模型说明 | Checkpoint | Inference Model |
| --- | --- | --- | ---|
| PP-HumanSeg-Lite | 超轻量级模型，适用于Web端或移动端实时分割场景，例如手机自拍、Web视频会议，模型结构为[Paddle自研模型](../../configs/pp_humanseg_lite/README.md)，推荐输入大小（398，224） | [lite_portrait_ckpt](https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz) | [lite_portrait_inference](https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz) |

#### 模型性能

| 模型名 |Input Size | FLOPs | Parameters | 计算耗时 | 模型大小 |
|-|-|-|-|-|-|
| PP-HumanSeg-Lite | 398x224 | 266M | 137K | 23.49ms | 543Kb |
| PP-HumanSeg-Lite | 288x162 | 138M | 137K | 15.62ms | 543Kb |

测试环境: 使用Paddle.js converter优化图结构，部署于Web端，显卡型号AMD Radeon Pro 5300M 4 GB。


## 安装

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。


#### 2. 安装PaddleSeg包

```shell
pip install paddleseg
```

#### 3. 下载PaddleSeg仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

## 快速体验
以下所有命令均在`PaddleSeg/contrib/PP-HumanSeg`目录下执行。
```shell
cd PaddleSeg/contrib/PP-HumanSeg
```

### 下载Inference Model

执行以下脚本快速下载所有Inference Model
```bash
python export_model/download_export_model.py
```

### 下载测试数据
我们提供了一些测试数据，从人像分割数据集 [Supervise.ly Person](https://app.supervise.ly/ecosystem/projects/persons) 中随机抽取一小部分并转化成PaddleSeg可直接加载数据格式，以下称为mini_supervisely，同时提供了手机前置摄像头的人像测试视频`video_test.mp4`。通过运行以下代码进行快速下载：

```bash
python data/download_data.py
```

### 视频流人像分割
```bash
# 通过电脑摄像头进行实时分割处理
python bg_replace.py \
--config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml

# 对人像视频进行分割处理
python bg_replace.py \
--config export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax/deploy.yaml \
--video_path data/video_test.mp4
```

视频分割结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/result.gif" width="20%" height="20%">

我们也支持使用 DIS（Dense Inverse Search-basedmethod）光流后处理算法，通过结合光流结果与分割结果，减少视频预测前后帧闪烁的问题。只要使用`--use_optic_flow`即可开启光流后处理，例如
```bash
# 增加光流后处理
python bg_replace.py \
--config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml \
--use_optic_flow
```

### 视频流背景替换
根据所选背景进行背景替换，背景可以是一张图片，也可以是一段视频。
```bash
# 通过电脑摄像头进行实时背景替换处理。可通过'--background_video_path'传入背景视频
python bg_replace.py \
--config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml \
--input_shape 224 398 \
--bg_img_path data/background.jpg

# 对人像视频进行背景替换处理。可通过'--background_video_path'传入背景视频
python bg_replace.py \
--config export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax/deploy.yaml \
--bg_img_path data/background.jpg \
--video_path data/video_test.mp4

# 对单张图像进行背景替换
python bg_replace.py \
--config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml \
--input_shape 224 398 \
--img_path data/human_image.jpg \
--bg_img_path data/background.jpg

```


背景替换结果如下：

<img src="https://paddleseg.bj.bcebos.com/humanseg/data/video_test.gif" width="20%" height="20%"><img src="https://paddleseg.bj.bcebos.com/humanseg/data/bg_replace.gif" width="20%" height="20%">


**NOTE**:

视频分割处理时间需要几分钟，请耐心等待。

Portrait模型适用于宽屏拍摄场景，竖屏效果会略差一些。

### 在线运行教程
我们提供了基于AI Studio的[在线运行教程](https://aistudio.baidu.com/aistudio/projectdetail/2189481)，方便您进行实践体验。

## 训练评估预测演示
如果上述大规模数据预训练的模型不能满足您的精度需要，可以基于上述模型在您的场景中进行Fine-tuning，以更好地适应您的使用场景。

### 下载预训练模型

执行以下脚本快速下载所有Checkpoint作为预训练模型
```bash
python pretrained_model/download_pretrained_model.py
```

### 训练
演示如何基于上述模型进行Fine-tuning。我们使用抽取的mini_supervisely数据集作为示例数据集，以PP-HumanSeg-Mobile为例，训练命令如下：
```bash
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python train.py \
--config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml \
--save_dir saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely \
--save_interval 100 --do_eval --use_vdl
```

更多命令行帮助可运行下述命令进行查看：
```bash
python train.py --help
```

### 评估
使用下述命令进行评估
```bash
python val.py \
--config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml \
--model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams
```

### 预测
使用下述命令进行预测， 预测结果默认保存在`./output/result/`文件夹中。
```bash
python predict.py \
--config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml \
--model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams \
--image_path data/human_image.jpg
```

## 模型导出
### 将模型导出为静态图模型

请确保位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python ../../export.py \
--config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml \
--model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams \
--save_dir export_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely_with_softmax \
--without_argmax --with_softmax
```

【注】模型导出时必须携带`--without_argmax --with_softmax`参数。

导出PP-HumanSeg-Lite模型：

```shell
python ../../export.py \
--config ../../configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml \
--save_dir export_model/pp_humanseg_lite_portrait_398x224_with_softmax \
--model_path pretrained_model/ppseg_lite_portrait_398x224/model.pdparams \
--without_argmax --with_softmax
```

### 导出脚本参数解释

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|配置文件|是|-|
|save_dir|模型和visualdl日志文件的保存根路径|否|output|
|model_path|预训练模型参数的路径|否|配置文件中指定值|
|with_softmax|在网络末端添加softmax算子。由于PaddleSeg组网默认返回logits，如果想要部署模型获取概率值，可以置为True|否|False|
|without_argmax|是否不在网络末端添加argmax算子。由于PaddleSeg组网默认返回logits，为部署模型可以直接获取预测结果，我们默认在网络末端添加argmax算子|否|False|

### 结果文件

```shell
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

## Web端部署

![image](https://user-images.githubusercontent.com/10822846/118273079-127bf480-b4f6-11eb-84c0-8a0bbc7c7433.png)

参见[Web端部署教程](../../deploy/web)

## 移动端部署

<img src="../../deploy/lite/example/human_1.png"  width="20%" >  <img src="../../deploy/lite/example/human_2.png"  width="20%" >  <img src="../../deploy/lite/example/human_3.png"  width="20%" >

参见[移动端部署教程](../../deploy/lite/)
