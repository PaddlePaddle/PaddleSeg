简体中文 | [English](README.md)

# Natural Image Matting
Image Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。


<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif" width="100%" height="100%">
</p>

# 快速体验
欢迎使用基于PP-Matting模型开发的在线抠图应用，“[懒人抠图](http://seg.itmanbu.com/)"。

<p align="center">
<img src="https://user-images.githubusercontent.com/48433081/165077834-c3191509-aeaf-45c8-b226-656174f4c152.gif" width="100%" height="100%">
</p>

## 更新动态
2022.07
【1】开源PPMatting代码。
【2】新增ClosedFormMatting、KNNMatting、FastMatting、LearningBaseMatting和RandomWalksMatting传统机器学习算法。
【3】新增GCA模型。
【4】完善目录结构。
【5】支持指定指标进行评估。

2022.04
【1】新增PPMatting模型。
【2】新增PPHumanMatting高分辨人像抠图模型。
【3】新增Grad、Conn评估指标。
【4】新增前景评估功能，利用[ML](https://arxiv.org/pdf/2006.14970.pdf)算法在预测和背景替换时进行前景评估。
【5】新增GradientLoss和LaplacianLoss。
【6】新增RandomSharpen、RandomSharpen、RandomReJpeg、RSSN数据增强策略。

2021.11 Matting项目开源, 实现图像抠图功能。
【1】支持Matting模型：DIM， MODNet。
【2】支持模型导出及Python部署。
【3】支持背景替换功能。
【4】支持人像抠图Android部署

## 目录
- [环境配置](#环境配置)
- [模型](#模型)
- [数据准备](#数据准备)
- [训练评估预测](#训练评估预测)
- [背景替换](#背景替换)
- [导出部署](#导出部署)
- [人像抠图Android部署](./deploy/human_matting_android_demo/README.md)


## 环境配置

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

#### 2. 下载PaddleSeg仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### 3. 安装

```shell
cd PaddleSeg/Matting
pip install -r requirements.txt
```

## 模型
提供多种场景人像抠图模型, 可根据实际情况选择相应模型，我们提供了Inference Model，您可直接下载进行[部署应用](#应用部署)。

模型推荐：
- 追求精度：PP-Matting, 低分辨率使用PP-Matting-512, 高分辨率使用PP-Matting-1024。
- 追求速度：ModNet-MobileNetV2。
- 高分辨率(>2048)简单背景人像抠图：PP-HumanMatting。
- 提供trimap：DIM-VGG16。

| 模型 | Params(M) | FLOPs(G) | FPS | Checkpoint | Inference Model |
| - | - | -| - | - | - |
| PP-Matting-512     | 24.5 | 91.28 | 28.9 | [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-matting-hrnet_w18-human_512.zip) |
| PP-Matting-1024    | 24.5 | 91.28 | 13.4(1024X1024) | [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_1024.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-matting-hrnet_w18-human_1024.zip) |
| PP-HumanMatting    | 63.9 | 135.8 (2048X2048)| 32.8(2048X2048)| [model](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-humanmatting-resnet34_vd.zip) |
| ModNet-MobileNetV2 | 6.5 | 15.7 | 68.4 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip) |
| ModNet-ResNet50_vd | 92.2 | 151.6 | 29.0 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-resnet50_vd.zip) |
| ModNet-HRNet_W18   | 10.2 | 28.5 | 62.6 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip) |
| DIM-VGG16          | 28.4 | 175.5| 30.4 | [model](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/dim-vgg16.zip) |

注意：FLOPs和FPS计算默认模型输入大小为(512, 512), GPU为Tesla V100 32G。

## 数据准备

利用MODNet开源的[PPM-100](https://github.com/ZHKKKe/PPM)数据集作为我们教程的示例数据集

将数据集整理为如下结构， 并将数据集置于data目录下。

```
PPM-100/
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha
|
|--train.txt
|
|--val.txt
```
其中，fg目录下的图象名称需和alpha目录下的名称一一对应

train.txt和val.txt的内容如下
```
train/fg/14299313536_ea3e61076c_o.jpg
train/fg/14429083354_23c8fddff5_o.jpg
train/fg/14559969490_d33552a324_o.jpg
...
```
可直接下载整理后的[PPM-100](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip)数据进行后续教程


如果完整图象需由前景和背景进行合成的数据集，类似[Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf)论文里使用的数据集Composition-1k，则数据集应整理成如下结构：
```
Composition-1k/
|--bg/
|
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|  |--trimap/ (如果存在)
|
|--train.txt
|
|--val.txt
```
train.txt的内容如下：
```
train/fg/fg1.jpg bg/bg1.jpg
train/fg/fg2.jpg bg/bg2.jpg
train/fg/fg3.jpg bg/bg3.jpg
...
```

val.txt的内容如下, 如果不存在对应的trimap，则第三列可不提供，代码将会自动生成。
```
val/fg/fg1.jpg bg/bg1.jpg val/trimap/trimap1.jpg
val/fg/fg2.jpg bg/bg2.jpg val/trimap/trimap2.jpg
val/fg/fg3.jpg bg/bg3.jpg val/trimap/trimap3.jpg
...
```

## 训练评估预测
### 训练
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
       --config configs/quick_start/modnet-mobilenetv2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --num_workers 5 \
       --save_dir output
```

**note:** 使用--do_eval会影响训练速度及增加显存消耗，根据需求进行开闭。
打开的时候会根据SAD保存历史最佳模型到`{save_dir}/best_model`下面，同时会在该目录下生成`best_sad.txt`记录下此时各个指标信息及iter.

`--num_workers` 多进程数据读取，加快数据预处理速度

更多参数信息请运行如下命令进行查看:
```shell
python tools/train.py --help
```
如需使用多卡，请用`python -m paddle.distributed.launch`进行启动

### 评估
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/val.py \
       --config configs/quick_start/modnet-mobilenetv2.yml \
       --model_path output/best_model/model.pdparams \
       --save_dir ./output/results \
       --save_results
```
`--save_result` 开启会保留图片的预测结果，可选择关闭以加快评估速度。

你可以直接下载我们提供的模型进行评估。

更多参数信息请运行如下命令进行查看:
```shell
python tools/val.py --help
```

### 预测
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --image_path data/PPM-100/val/fg/ \
    --save_dir ./output/results \
    --fg_estimate True
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

`--fg_estimate False` 可关闭前景估计功能，可提升预测速度，但图像质量会有所降低

你可以直接下载我们提供的模型进行预测。

更多参数信息请运行如下命令进行查看:
```shell
python tools/predict.py --help
```


## 背景替换
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --image_path path/to/your/image \
    --background path/to/your/background/image \
    --save_dir ./output/results \
    --fg_estimate True
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

`--background`可以传入背景图片路劲，或选择（'r','g','b','w')中的一种，代表红，绿，蓝，白背景, 若不提供则采用绿色作为背景。

`--fg_estimate False` 可关闭前景估计功能，可提升预测速度，但图像质量会有所降低

**注意：** `--image_path`必须是一张图片的具体路径。

你可以直接下载我们提供的模型进行背景替换。

更多参数信息请运行如下命令进行查看:
```shell
python tools/bg_replace.py --help
```

## 导出部署
### 模型导出
```shell
python tools/export.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --save_dir output/export
```
如果模型（比如：DIM）需要trimap的输入，需要增加参数`--trimap`

更多参数信息请运行如下命令进行查看:
```shell
python tools/export.py --help
```

### 应用部署
```shell
python deploy/python/infer.py \
    --config output/export/deploy.yaml \
    --image_path data/PPM-100/val/fg/ \
    --save_dir output/results \
    --fg_estimate True
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

`--fg_estimate False` 可关闭前景估计功能，可提升预测速度，但图像质量会有所降低

更多参数信息请运行如下命令进行查看:
```shell
python deploy/python/infer.py --help
```

## 致谢

* 感谢[钱彬(Qianbin)](https://github.com/qianbin1989228)等开发者的贡献。
* 感谢Jizhizi Li等提出的[GFM](https://arxiv.org/abs/2010.16188) Matting框架助力PP-Matting的算法研发。
