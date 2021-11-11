# LaneSeg 模型训练教程

* 本教程旨在介绍如何通过使用PaddleSeg进行车道线检测

* 在阅读本教程前，请确保您已经了解过PaddleSeg的[快速入门](../../README.md#快速入门)和[基础功能](../../README.md#基础功能)等章节，以便对PaddleSeg有一定的了解

## 环境依赖

* PaddlePaddle >= 2.1.2 或develop版本
* Python 3.6+


## 一. 准备待训练数据


在这个[页面](https://github.com/TuSimple/tusimple-benchmark/issues/3)下载原始数据集。通过以下代码执行生成。

```shell
python tools/generate_seg_tusimple.py.py --root path/to/your/unzipped/file

```

解压得到的train_set和test_set数据，组织成如下目录结构
```
LaneSeg
|-- data
    |-- tusimple
        |-- clips
            |-- 0313-1
            |-- 0313-2
            |-- 0530
            |-- 0531
            |-- 0601
        |-- label_data_0313.json
        |-- label_data_0531.json
        |-- label_data_0601.json
        |-- test_label.json
        |-- test_tasks_0627.json
```

## 二. 准备配置

接着我们需要确定相关配置，从本教程的角度，配置分为三部分：

* 数据集
  * 数据集类型
  * 数据集根目录
  * 数据增强
* 损失函数
  * 损失函数类型
  * 损失函数系数
* 其他
  * 学习率
  * Batch大小
  * ...

数据集，包括训练数据集和验证数据集，数据集的配置和数据路径有关，在本教程中，数据存放在`data/tusimple`中

其他配置则根据数据集和机器环境的情况进行调节，最终我们保存一个如下内容的yaml配置文件，存放路径为**configs/lane_tusimple_seg.yml.yaml**

```yaml
batch_size: 4
iters: 320000

train_dataset:
  type: TusimpleSeg
  dataset_root: ./data/tusimple/
  cut_height: 160
  transforms:
    - type: LaneRandomRotation
    - type: RandomHorizontalFlip
    - type: Resize
      target_size: [640, 368]
    - type: RandomDistort
      brightness_range: 0.25
      brightness_prob: 1
      contrast_range: 0.25
      contrast_prob: 1
      saturation_range: 0.25
      saturation_prob: 1
      hue_range: 63
      hue_prob: 1
    - type: Normalize
  mode: train

val_dataset:
  type: TusimpleSeg
  dataset_root: ./data/tusimple/
  cut_height: 160
  transforms:
    - type: Resize
      target_size: [640, 368]
    - type: Normalize
  mode: val

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.001
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: LaneCrossEntropyLoss
  coef: [1]

model:
  type: BiSeNetLane
  pretrained: Null

```


## 三. 开始训练

使用下述命令启动训练

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡

**windows下请执行以下命令**
**set CUDA_VISIBLE_DEVICES=0**
python train.py \
       --config configs/lanenet.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```
多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1 # 设置2张可用的卡
python -m paddle.distributed.launch train.py \
       --config configs/lanenet.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

恢复训练

```shell
python train.py \
       --config configs/lanenet.yml \
       --resume_model output/iter_20000 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

## 四. 进行评估

模型训练完成，使用下述命令启动评估

```shell
python val.py \
       --config configs/lanenet.yml \
       --model_path output/iter_20000/model.pdparams

```

## 五. 可视化

```shell
python predict.py \
       --config configs/lanenet.yml \
       --model_path output/iter_20000/model.pdparams \
       --image_path data/test_images/0.jpg \
       --save_dir output/result

```
