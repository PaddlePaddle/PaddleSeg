# 全流程开发

## 目录
* [环境配置](#环境配置)
* [示例数据集](#示例数据集)
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


## 背景替换
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

## 导出部署
### 模型导出
```shell
python tools/export.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
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

## 备注
该教程适用于[configs](../configs)下的所有模型。
