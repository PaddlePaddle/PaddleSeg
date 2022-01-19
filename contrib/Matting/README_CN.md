简体中文 | [English](README.md)

# Matting
Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。


<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/141714637-be8af7b1-ccd0-49df-a4f9-10423705802e.jpg" width="100%" height="100%">
</p>

## 更新动态
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
cd PaddleSeg
pip install -e .
pip install scikit-image
cd contrib/Matting
```

## 模型

[PP-HumanMatting](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams)

[DIM-VGG16](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams)

MODNet在[PPM-100](https://github.com/ZHKKKe/PPM)数据集上的性能

| Backbone | SAD | MSE | Params(M) | FLOPs(G) | FPS | Link |
|-|-|-|-|-|-|-|
|MobileNetV2|112.73|0.0098|6.5|15.7|67.5|[model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams)|
|ResNet50_vd|104.14|0.0090|92.2|151.6|28.6|[model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams)|
|HRNet_W18|77.96|0.0054|10.2|28.5|10.9|[model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams)|

注意：模型输入大小为(512, 512), GPU为Tesla V100 32G。

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
python train.py \
       --config configs/modnet/modnet_mobilenetv2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 5000 \
       --num_workers 5 \
       --save_dir output
```

**note:** 使用--do_eval会影响训练速度及增加显存消耗，根据需求进行开闭。

`--num_workers` 多进程数据读取，加快数据预处理速度

更多参数信息请运行如下命令进行查看:
```shell
python train.py --help
```
如需使用多卡，请用`python -m paddle.distributed.launch`进行启动

### 评估
```shell
export CUDA_VISIBLE_DEVICES=0
python val.py \
       --config configs/modnet/modnet_mobilenetv2.yml \
       --model_path output/best_model/model.pdparams \
       --save_dir ./output/results \
       --save_results
```
`--save_result` 开启会保留图片的预测结果，可选择关闭以加快评估速度。

你可以直接下载我们提供的模型进行评估。

更多参数信息请运行如下命令进行查看:
```shell
python val.py --help
```

### 预测
```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py \
    --config configs/modnet/modnet_mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --image_path data/PPM-100/val/fg/ \
    --save_dir ./output/results
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

你可以直接下载我们提供的模型进行预测。

更多参数信息请运行如下命令进行查看:
```shell
python predict.py --help
```


## 背景替换
```shell
export CUDA_VISIBLE_DEVICES=0
python bg_replace.py \
    --config configs/modnet/modnet_mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --image_path path/to/your/image \
    --bg_path path/to/your/background/image \
    --save_dir ./output/results
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

若不提供`--bg_path`, 则采用绿色作为背景。

**注意：** `--image_path`必须是一张图片的具体路径。

你可以直接下载我们提供的模型进行背景替换。

更多参数信息请运行如下命令进行查看:
```shell
python bg_replace.py --help
```

## 导出部署
### 模型导出
```shell
python export.py \
    --config configs/modnet/modnet_mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --save_dir output/export
```
如果模型（比如：DIM）需要trimap的输入，需要增加参数`--trimap`

更多参数信息请运行如下命令进行查看:
```shell
python export.py --help
```

### 应用部署
```shell
python deploy/python/infer.py \
    --config output/export/deploy.yaml \
    --image_path data/PPM-100/val/fg/ \
    --save_dir output/results
```
如模型需要trimap信息，需要通过`--trimap_path`传入trimap路径。

更多参数信息请运行如下命令进行查看:
```shell
python deploy/python/infer.py --help
```

## 贡献者

感谢
[wuyefeilin](https://github.com/wuyefeilin)、
[钱彬(Qianbin)](https://github.com/qianbin1989228)、
[yzl19940819](https://github.com/yzl19940819)
等开发者的贡献
