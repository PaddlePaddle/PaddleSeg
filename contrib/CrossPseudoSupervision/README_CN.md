简体中文 | [English](README.md)

# CPS: 基于交叉伪监督的半监督语义分割

不同于图像分类任务，**数据的标注对于语义分割任务来说是比较困难而且成本高昂的**，需要为图像的每一个像素标注一个标签，包括一些特别细节的物体，如电线杆等。但是对于获取RGB数据是比较简单，如何**利用大量的无标注数据去提高模型的性能**，便是半监督语义分割领域研究的问题。

[Cross pseudo supervision, CPS](https://arxiv.org/abs/2106.01226)是一种**非常简洁而又性能很好**的半监督语义分割任务算法。训练时，使用两个相同结构、但是不同初始化的网络，添加约束**使得两个网络对同一样本的输出是相似的**。具体来说，当前网络产生的one-hot pseudo label，会作为另一路网络预测的目标，这个过程可以用cross entropy loss监督，就像传统的全监督语义分割任务的监督一样。**该算法在在两个benchmark (PASCAL VOC, Cityscapes) 都取得了SOTA的结果**

部分可视化结果如下，左边为RGB图像，中间为预测图，右边为真值

![](https://ai-studio-static-online.cdn.bcebos.com/e45cf14fe5a0417791a5455e84b2243abe8d8298cba94c8f9d98b0edd8a431d1)

![](https://ai-studio-static-online.cdn.bcebos.com/a136cff0ce7748dbb7bd3d8fa5b6942b8eeee8cb10ec456694c749a09375fc29)


## 目录
- [环境配置](#环境配置)
- [模型](#模型)
- [数据准备](#数据准备)
- [训练评估预测](#训练评估预测)

## 环境配置


- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
    - 版本要求：PaddlePaddle develop, Python>=3.7

- PaddleSeg安装，通过以下命令：

```shell
git clone -b develop https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

## 模型

本项目复现的为在Cityscapes数据集以0.5倍数据量上的结果，复现模型的mIOU为**78.39%**，对比如下表所示：

| CPS.resnet50.deeplabv3+(1/2 Cityscapes) | mIOU |
| --- | --- |
| pytorch | 78.77% |
| paddle | 78.39% |

Paddle复现的模型权重下载链接为：

## 数据准备

使用CPS源代码所提供的Cityscapes数据集，已上传至[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/177911)，建议将数据解压至`contrib/CrossPseudoSupervision/data`文件夹下，准备好的数据组织如下：

```
data/
|-- city
    ├── config_new
    │    ├── coarse_split
    │    │   ├── train_extra_3000.txt
    │    │   ├── train_extra_6000.txt
    │    │   └── train_extra_9000.txt
    │    ├── subset_train
    │    │   ├── train_aug_labeled_1-16.txt
    │    │   ├── train_aug_labeled_1-2.txt
    │    │   ├── train_aug_labeled_1-4.txt
    │    │   ├── train_aug_labeled_1-8.txt
    │    │   ├── train_aug_unlabeled_1-16.txt
    │    │   ├── train_aug_unlabeled_1-2.txt
    │    │   ├── train_aug_unlabeled_1-4.txt
    │    │   └── train_aug_unlabeled_1-8.txt
    │    ├── test.txt
    │    ├── train.txt
    │    ├── train_val.txt
    │    └── val.txt  
    ├── generate_colored_gt.py
    ├── images
    │   ├── test
    │   ├── train
    │   └── val
    └── segmentation
        ├── test
        ├── train
        └── val
```

## 训练评估预测

执行以下命令，进入到`CPS`所在分支：

```shell
cd ./contrib/CrossPseudoSupervision
```

### 训练

准备好环境与数据之后，执行以下命令训练：

```shell
python train.py --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml --log_iters 10 --save_dir ./output/ --batch_size 2
```

建议使用单机多卡进行训练，执行以下命令启动四卡训练：

```shell
python -m paddle.distributed.launch --gpus="0,1,2,3" train.py --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
--log_iters 10 --save_dir $SAVE_PATH$ --batch_size 8
```

**注**： 论文原repo中训练过程并不验证，而是训练结束后再对后几个保存的权重逐个测试精度，因此本项目也不在训练过程中验证；配置文件是训练1/2有标签的数据，若要改为其他比例，训练epoch数需要进行调整

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| Cityscapes | 128  | 137  | 160  | 240  |


### 评估

训练结束后，执行以下命令评估模型精度：

```shell
export CUDA_VISIBLE_DEVICES=0
python val.py \
       --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
       --model_path $MODEL_PATH$
```

### 预测

```shell
export CUDA_VISIBLE_DEVICES=0
!python predict.py \
       --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
       --model_path $MODEL_PATH$ \
       --image_path $IMG_PATH$ \
       --save_dir $SAVE_PATH$ \
       --is_slide \
       --crop_size 800 800 \
       --stride 532 532
```

你可以直接下载我们提供的模型进行预测。
