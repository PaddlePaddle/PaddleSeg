简体中文 | [English](README.md)

# 人像分割PP-HumanSeg

## 目录
- [简介](#简介)
- [最新消息](#最新消息)
- [PP-HumanSeg模型](#PP-HumanSeg模型)
  - [肖像分割模型](#肖像分割模型)
  - [通用人像分割模型](#通用人像分割模型)
- [安装](#安装)
- [快速体验](#快速体验)
  - [视频流人像分割](#视频流人像分割)
  - [视频流背景替换](#视频流背景替换)
  - [在线运行教程](#在线运行教程)
- [训练评估预测演示](#训练评估预测演示)
- [模型导出](#模型导出)
- [Web端部署](#Web端部署)
- [移动端部署](#移动端部署)

## 简介

将人物和背景在像素级别进行区分，是一个图像分割的经典任务，具有广泛的应用。
一般而言，该任务可以分为两类：1）针对半身人像的分割，简称肖像分割；2）针对全身和半身人像的分割，简称通用人像分割。

对于肖像分割和通用人像分割，PaddleSeg发布了PP-HumanSeg系列模型:
* 分割精度高、推理速度快、通用型强
* 可以开箱即用，零成本部署到产品中，也可以针对特定场景数据进行微调，实现更佳分割效果

大家可以在Paddle.js的网页体验[人像扣图](https://paddlejs.baidu.com/humanseg)效果、[视频背景替换及弹幕穿透](https://www.paddlepaddle.org.cn/paddlejs)效果。

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/149886667-f47cab88-e81a-4fd7-9f32-fbb34a5ed7ce.png"  height="200">        <img src="https://user-images.githubusercontent.com/30695251/149887482-d1fcd5d3-2cce-41b5-819b-bfc7126b7db4.png"  height="200">
</p>

## 最新消息
- [2022-7] 发布PP-HumanSeg V2版本模型，肖像分割模型的推理速度提升45.5%、mIoU提升0.63%、可视化效果更佳，通用人像分割模型的推理速度提升xx，mIoU提升xx。
- [2022-1] 人像分割论文[PP-HumanSeg](./paper.md)发表于WACV 2022 Workshop，并开源连通性学习（SCL）方法和大规模视频会议数据集。
- [2021-7] 百度视频会议可实现Web端一秒入会，其中的虚拟背景功能采用我们的PP-HumanSeg肖像模型，实现实时背景替换和背景虚化功能，保护用户隐私，并增加视频会议的趣味性。
- [2021-7] 发布PP-HumanSeg V1版本模型，包括一个肖像分割模型和两个通用人像分割模型。

<p align="center">
<img src="https://github.com/LutaoChu/transfer_station/raw/master/conference.gif" width="40%" height="40%">
</p>

## PP-HumanSeg模型

### 肖像分割模型

PP-HumanSeg肖像分割模型，适用于手机视频通话、Web视频会议等实时分割场景，可以开箱即用，免去大家标注数据、训练模型的成本。

PP-HumanSeg-Lite-V1肖像分割模型使用[自研模型结构](../../configs/pp_humanseg_lite/README.md)，分割效果较好，模型体积非常小。

PP-HumanSeg-Lite-V2肖像分割模型，对比V1模型，推理速度提升45.5%、mIoU提升0.63%、可视化效果更佳，核心在于：
* 更高的分割精度：使用PaddleSeg推出的[超轻量级分割模型](../../configs/mobileseg/README.md)，具体选择MobileNetV3作为骨干网络，设计多尺度特征融合模块(Multi-Scale Feature Aggregation Module)。
* 更快的推理速度：减小模型最佳输入尺寸，既减少了推理耗时，又增大模型感受野。
* 更好的通用性：使用迁移学习的思想，首先在大型通用人像分割数据集上预训练，然后在小型肖像分割数据集上微调。

V1和V2肖像分割模型的具体信息如下表格。

| 模型名 | 最佳输入尺寸 | mIou(%) | 推理耗时(ms) | 模型体积(MB) | 配置文件 | Checkpoint | Inference Model |
| --- | --- | --- | ---| --- | --- | --- | ---|
| PP-HumanSeg-Lite-V1 | 398x224 | 96.00 | 29.68 | 2.2 | [cfg](./configs/portrait14k_pp_humanseg_lite_v1.yml) | [url](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humanseg_lite_v1_398x224_pretrained.zip) | [url](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humanseg_lite_v1_398x224_inference_model.zip) |
| PP-HumanSeg-Lite-V2 | 256x144 | 96.63 | 15.86 | 13.5 | [cfg](./configs/portrait14k_pp_humanseg_lite_v2.yml) | [url](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humanseg_lite_v2_256x144_pretrained.zip) | [url](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humanseg_lite_v2_256x144_inference_model.zip) |

说明：
* 测试肖像模型的精度mIoU：针对PP-HumanSeg-14k数据集，使用模型最佳输入尺寸进行测试，没有应用多尺度和flip等操作。
* 测试肖像模型的推理耗时：基于PaddleLite预测库，小米9手机（骁龙855 CPU）、单线程、大核，使用模型最佳输入尺寸进行测试。
* 在手机端部署肖像分割模型，可能存在横屏和竖屏等情况。大家可以根据实际情况对图像进行旋转，保持人像是竖直情况，然后将图像（尺寸为256x144或144x256）输入模型，可以得到最佳的分割效果。

### 通用人像分割模型

针对通用人像分割任务，PP-HumanSeg开放了在大规模人像数据上训练的三个人像模型，满足服务端、移动端、Web端多种使用场景的需求。

| 模型名 | 模型说明 | Checkpoint | Inference Model |
| --- | --- | --- | ---|
| PP-HumanSeg-Server | 高精度模型，适用于服务端GPU且背景复杂的场景， 模型结构为Deeplabv3+/ResNet50, 输入大小（512， 512） |[server_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip) | [server_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.zip) |
| PP-HumanSeg-Mobile | 轻量级模型，适用于移动端或服务端CPU的前置摄像头场景，模型结构为HRNet_w18_samll_v1，输入大小（192， 192）  | [mobile_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip) | [mobile_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/fcn_hrnetw18_small_v1_humanseg_192x192_with_softmax.zip) |
| PP-HumanSeg-Lite | 超轻量级模型，适用于Web端或移动端实时分割场景，例如手机自拍、Web视频会议，模型结构为[Paddle自研模型](../../configs/pp_humanseg_lite/README.md)，输入大小（192， 192） | [lite_ckpt](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/pphumanseg_lite_generic_192x192.zip) | [lite_inference](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/pphumanseg_lite_generic_192x192_with_softmax.zip) |


| 模型名 | 最佳输入尺寸 | mIou(%) | 推理耗时(ms) | 模型体积(MB) | 配置文件 | Checkpoint | Inference Model |
| --- | --- | --- | ---| --- | --- | --- | ---|
| PP-HumanSeg-Lite-V1   | 192x192 | 91.61
| PP-HumanSeg-Lite-V2   | 192x192 | 93.75
| PP-HumanSeg-Mobile-V1 | 192x192 | 94.06
| PP-HumanSeg-Mobile-V2 | 192x192 | 94.67
| PP-HumanSeg-Server-V1 | 192x192 | -



NOTE:
* 其中Checkpoint为模型权重，用于Fine-tuning场景。

* Inference Model为预测部署模型，包含`model.pdmodel`计算图结构、`model.pdiparams`模型参数和`deploy.yaml`基础的模型配置信息。

* 其中Inference Model适用于服务端的CPU和GPU预测部署，适用于通过Paddle Lite进行移动端等端侧设备部署。更多Paddle Lite部署说明查看[Paddle Lite文档](https://paddle-lite.readthedocs.io/zh/latest/)

模型性能：

| 模型名 |Input Size | FLOPs | Parameters | 计算耗时 | 模型大小 |
|-|-|-|-|-|-|
| PP-HumanSeg-Server | 512x512 | 114G | 26.8M | 37.96ms | 103Mb |
| PP-HumanSeg-Mobile | 192x192 | 584M | 1.54M | 13.17ms | 5.9Mb |
| PP-HumanSeg-Lite | 192x192 | 121M | 137K | 10.51ms | 543Kb |

测试环境：Nvidia Tesla V100单卡。


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
--config configs/pp_humanseg_mobile_192x192_mini_supervisely.yml \
--save_dir saved_model/pp_humanseg_mobile_192x192_mini_supervisely \
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
--config configs/pp_humanseg_mobile_192x192_mini_supervisely.yml \
--model_path saved_model/pp_humanseg_mobile_192x192_mini_supervisely/best_model/model.pdparams
```

### 预测
使用下述命令进行预测， 预测结果默认保存在`./output/result/`文件夹中。
```bash
python predict.py \
--config configs/pp_humanseg_mobile_192x192_mini_supervisely.yml \
--model_path saved_model/pp_humanseg_mobile_192x192_mini_supervisely/best_model/model.pdparams \
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
--config configs/pp_humanseg_mobile_192x192_mini_supervisely.yml \
--model_path saved_model/pp_humanseg_mobile_192x192_mini_supervisely/best_model/model.pdparams \
--save_dir export_model/pp_humanseg_mobile_192x192_mini_supervisely_with_softmax \
--without_argmax --with_softmax
```

【注】这里采用软预测结果，可以携带透明度，使得边缘更为平滑。因此模型导出时必须携带`--without_argmax --with_softmax`参数。

导出PP-HumanSeg-Lite模型：

```shell
python ../../export.py \
--config ../../configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml \
--save_dir export_model/pp_humanseg_lite_portrait_398x224_with_softmax \
--model_path pretrained_model/ppseg_lite_portrait_398x224/model.pdparams \
--without_argmax --with_softmax
```

其他PP-HumanSeg模型对应的导出yml位于`configs/`和`../../configs/pp_humanseg_lite/`目录下。

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
