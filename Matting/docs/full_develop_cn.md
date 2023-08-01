# 全流程开发

## 目录
* [环境配置](#环境配置)
* [数据集准备](#数据集准备)
* [模型选择](#模型选择)
* [训练](#训练)
* [评估](#评估)
* [预测](#预测)
* [背景替换](#背景替换)
* [导出部署](#导出部署)

## 环境配置

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

由于图像抠图模型计算开销大，推荐在GPU版本的PaddlePaddle下使用。
推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

#### 2. 下载PaddleSeg仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### 3. 安装

```shell
cd PaddleSeg/Matting
pip install "paddleseg>=2.5"
pip install -r requirements.txt
```


## 数据集准备

利用MODNet开源的[PPM-100](https://github.com/ZHKKKe/PPM)数据集作为我们教程的示例数据集。自定已数据集请参考[数据集准备](data_prepare_cn.md)。


下载已经准备好的PPM-100数据集：
```shell
mkdir data && cd data
wget https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip
unzip PPM-100.zip
cd ..
```

数据集结构目录如下：

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

**注意** : 该数据集仅仅作为教程演示，无法利用其训练得到一个收敛的模型。

## 模型选择

Matting项目支持配置化直接驱动，模型配置文件均放置于[configs](../configs/)目录下，大家可根据实际情况选择相应的配置文件进行训练、预测等流程。Trimap-based类方法（DIM）暂不支持处理视频。

该教程中使用[configs/quick_start/ppmattingv2-stdc1-human_512.yml](../configs/quick_start/ppmattingv2-stdc1-human_512.yml)模型配置文件进行教学演示。


## 训练

```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
       --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
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

## 微调
如果想利用预训练模型进行微调（finetune），可以在配置文件中添加model.pretained字段，内容为预训练模型权重文件的URL地址或本地路径。下面以使用官方提供的PP-MattingV2模型进行微调为例进行说明。

首先进行预训练模型的下载。
下载[模型库](../README_CN.md/#模型库)中的预训练模型并放置于pretrained_models目录下。
```shell
mkdir pretrained_models && cd pretrained_models
wget https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams
cd ..
```
然后修改配置文件中的`train_dataset.dataset_root`、`val_dataset.dataset_root`、`model.pretrained`等字段，可适当降低学习率，其余字段保持不变即可。
```yaml
train_dataset:
  type: MattingDataset
  dataset_root: path/to/your/dataset # 自定义数据集路径
  mode: train

val_dataset:
  type: MattingDataset
  dataset_root: path/to/your/dataset # 自定义数据集路径
  mode: val

model:
  type: PPMattingV2
  backbone:
    type: STDC1
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz
  decoder_channels: [128, 96, 64, 32, 16]
  head_channel: 8
  dpp_output_channel: 256
  dpp_merge_type: add
  pretrained: pretrained_models/ppmattingv2-stdc1-human_512.pdparams # 刚刚下载的预训练模型文件
lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.001  # 可适当降低学习率
  end_lr: 0
  power: 0.9
  warmup_iters: 1000
  warmup_start_lr: 1.0e-5
```
接下来即可参考`训练`章节内容进行模型微调训练。

## 评估
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/val.py \
       --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
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

## 预测

### 图像预测
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
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

### 视频预测
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --video_path path/to/video \
    --save_dir ./output/results \
    --fg_estimate True
```


## 背景替换
### 图像背景替换
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
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
### 视频背景替换
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --video_path path/to/video \
    --background 'g' \
    --save_dir ./output/results \
    --fg_estimate True
```


## 导出部署
### 模型导出
```shell
python tools/export.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --save_dir output/export \
    --input_shape 1 3 512 512
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

`--video_path` 传入视频路径，可进行视频抠图

更多参数信息请运行如下命令进行查看:
```shell
python deploy/python/infer.py --help
```
