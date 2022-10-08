[English](distributed_train.md) | 简体中文


# 分布式训练

## 1. 简介

分布式训练指的是将训练任务按照一定方法拆分到多个计算节点进行计算，再按照一定的方法对拆分后计算得到的梯度等信息进行聚合与更新，起到加快训练速度的效果。

飞桨分布式训练技术源自百度的业务实践，在自然语言处理、计算机视觉、搜索和推荐等领域经过超大规模业务检验。分布式训练的高性能，是飞桨的核心优势技术之一。

PaddleSeg同时支持单机多卡训练与多机多卡训练。更多关于分布式训练的方法与文档可以参考：[分布式训练快速开始教程](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。

## 2. 使用方法

### 2.1 单机多卡训练

以PP-LiteSeg为例，本地准备好数据之后，使用`paddle.distributed.launch`或者`fleetrun`的接口启动训练任务即可。下面为运行脚本示例。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k
python -m paddle.distributed.launch \
    --log_dir=./log/ \
    --gpus "0,1,2,3,4,5,6,7" \
    train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 1000 \
    --num_workers 3 \
    --do_eval \
    --use_vdl
```

### 2.2 多机多卡训练

相比单机多卡训练，多机多卡训练时，只需要添加`--ips`的参数，该参数表示需要参与分布式训练的机器的ip列表，不同机器的ip用逗号隔开。下面为运行代码示例。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k
ip_list="a1.b1.c1.d1,a2.b2.c2.d2"
python -m paddle.distributed.launch \
    --log_dir=./log/ \
    --ips=${ip_list} \
    --gpus "0,1,2,3,4,5,6,7" \
    train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 1000 \
    --num_workers 3 \
    --do_eval \
    --use_vdl
```

**注意：**

* 不同机器的ip信息需要用逗号隔开，可以通过`ifconfig`或者`ipconfig`查看。
* 不同机器之间需要做免密设置，且可以直接ping通，否则无法完成通信。
* 不同机器之间的代码、数据与运行命令或脚本需要保持一致，且所有的机器上都需要运行设置好的训练命令或者脚本。最终`ip_list`中的第一台机器的第一块设备是trainer0，以此类推。
* 不同机器的起始端口可能不同，建议在启动多机任务前，在不同的机器中设置相同的多机运行起始端口，命令为`export FLAGS_START_PORT=17000`，端口值建议在`10000~20000`之间。


## 3. 性能效果测试

在3机8卡V100的机器上进行模型训练，不同模型的精度、训练耗时、多机加速比情况如下所示。

| 模型    | 骨干网络 | 数据集 | 配置   | 单机8卡耗时/精度 | 3机8卡耗时/精度 | 加速比  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  OCRNet | HRNet_w18 | Cityscapes | [ocrnet_hrnetw18_cityscapes_1024x512_160k.yml](../../configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml)  | 8.9h/80.91% | 5.33h/80.13%  | **1.88** |
|  SegFormer_B0 | - | Cityscapes | [segformer_b0_cityscapes_1024x1024_160k.yml](../../configs/segformer/segformer_b0_cityscapes_1024x1024_160k.yml)  | 5.61h/76.73% | 2.6h/75.86%  | **2.15** |
|  SegFormer_B0<sup>*</sup> | - | Cityscapes | [segformer_b0_cityscapes_1024x1024_160k.yml](../../configs/segformer/segformer_b0_cityscapes_1024x1024_160k.yml)  | 5.61h/76.73% | 3.5h/76.48%  | **1.60** |


在4机8卡V100的机器上进行模型训练，不同模型的精度、训练耗时、多机加速比情况如下所示。

| 模型    | 骨干网络 | 数据集 | 配置   | 单机8卡耗时/精度 | 4机8卡耗时/精度 | 加速比  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  PP-LiteSeg-T  | STDC1 | Cityscapes | [pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml](../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml)  | 7.58h/73.05% | 2.5h/72.43%%  | **3.03** |


**注意：**

* 在训练的GPU卡数过多时，精度会稍微有所损失（1%左右），此时可以尝试通过添加warmup或者适当增加迭代轮数来弥补精度损失。
* SegFormer_B0<sup>*</sup>表示增加SegFormer_B0模型的训练迭代轮数（默认配置在3机8卡训练时，迭代轮数为26.7k iter，这里增加至35k iter）。
