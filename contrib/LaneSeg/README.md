# LaneSeg 模型训练教程

* 车道线检测是属于自动驾驶算法范畴的一部分，可以用来辅助进行车辆定位和进行决策，本教程旨在介绍如何通过使用PaddleSeg进行车道线检测



* 在阅读本教程前，请确保您已经了解过PaddleSeg的[快速入门](../../README.md#快速入门)和[基础功能](../../README.md#基础功能)等章节，以便对PaddleSeg有一定的了解

## 环境依赖

* PaddlePaddle >= 2.1.2 或develop版本
* Python 3.6+


## 一. 准备待训练数据


在这个[页面](https://github.com/TuSimple/tusimple-benchmark/issues/3)下载原始数据集。通过以下代码执行生成。

```shell
python utils/generate_seg_tusimple.py.py --root path/to/your/unzipped/file

```

解压得到的train_set和test_set数据，组织成如下目录结构
```
 The folder structure is as follow:

 LaneSeg
 |-- data
     |-- tusimple
        |-- train_set
            |-- clips
                |-- 0313-1
                |-- 0313-2
                |-- 0531
                |-- 0601
            |-- labels [need to generate label dir]
                |-- 0313-1
                |-- 0313-2
                |-- 0531
                |-- 0601
            |-- train_list.txt [need to generate]
            |-- label_data_0313.json
            |-- label_data_0531.json
            |-- label_data_0601.json
        |-- test_set
            |-- clips
                |-- 0530
                |-- 0531
                |-- 0601
            |-- labels [need to generate label dir]
                |-- 0530
                |-- 0531
                |-- 0601
            |-- train_list.txt [need to generate]
            |-- test_tasks_0627.json
            |-- test_label.json
```

## 二. 开始训练

使用下述命令启动训练

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡

**windows下请执行以下命令**
**set CUDA_VISIBLE_DEVICES=0**
python train.py \
       --config configs/bisenet_tusimple_640x368_300k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 2000 \
       --save_dir output
```
多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1 # 设置2张可用的卡
python -m paddle.distributed.launch train.py \
       --config configs/bisenet_tusimple_640x368_300k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 2000 \
       --save_dir output
```

恢复训练

```shell
python train.py \
       --config configs/bisenet_tusimple_640x368_300k.yml \
       --resume_model output/iter_20000 \
       --do_eval \
       --use_vdl \
       --save_interval 2000 \
       --save_dir output
```

## 三. 进行评估

模型训练完成，使用下述命令启动评估

```shell
python val.py \
       --config configs/bisenet_tusimple_640x368_300k.yml \
       --model_path output/best_model/model.pdparams
```

## 四. 可视化

```shell
python predict.py \
       --config configs/bisenet_tusimple_640x368_300k.yml \
       --model_path output/best_model/model.pdparams \
       --image_path data/test_images/0.jpg \
       --save_dir output/result
```
可视化结果示例：

  预测结果：<br/>
  ![](data/images/points/3.jpg)<br/>
  分割结果：<br/>
  ![](data/images/pseudo_color_prediction/3.png)<br/>
  车道线预测结果：<br/>
  ![](data/images/added_prediction/3.jpg)

## 五. 模型导出

```shell
python export.py \
       --config configs/quick_start/bisenet_tusimple_640x368_300k.yml \
       --model_path output/best_model/model.pdparams
```

## 六. 应用部署

模型的部署，完成，使用下述命令启动评估

```shell
#运行如下命令，会在output文件下面生成一张3.jpg的图像
python deploy/infer.py \
--config output/deploy.yaml\
--image_path data/test_images/3.jpg\
--save_dir output
```
