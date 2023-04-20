简体中文 | [English](README.md)

# CPS: 基于交叉伪监督的半监督语义分割

不同于图像分类任务，**数据的标注对于语义分割任务来说是比较困难且成本高昂的**。图像中的每个像素都需要有一个标签，包括一些特别细节的物体，如电线杆等。与对像素的密集标注相比，获取原始RGB数据相对简单。因此，**如何利用大量的无标注数据提升模型的性能，是半监督语义分割领域的研究热点**。

[Cross pseudo supervision, CPS](https://arxiv.org/abs/2106.01226)是一种**简洁而高性能**的半监督语义分割任务算法。在训练时，使用两个相同结构、但是初始化状态不同的网络，添加约束**使得两个网络对同一样本的输出是相似的**。具体来说，一个网络生成的one-hot伪标签将作为训练另一个网络的目标。这个过程可以用交叉熵损失函数监督，就像传统的监督学习语义分割任务的一样。**该算法在在两个benchmark (PASCAL VOC, Cityscapes) 都取得了最先进的结果**。

部分可视化结果如下（左边为RGB图像，中间为预测图，右边为真值）:

![](https://user-images.githubusercontent.com/52785738/229003524-103fb081-dd36-4b19-b070-156d58467fe2.png)

![](https://user-images.githubusercontent.com/52785738/229003602-05cb2be1-8224-4600-8f6a-1ec58b909e47.png)



## 目录
- [环境配置](#环境配置)
- [模型](#模型)
- [数据准备](#数据准备)
- [训练评估预测](#训练评估预测)

## 环境配置


- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
    - 版本要求：PaddlePaddle develop (Nightly build), Python>=3.7

- PaddleSeg安装，通过以下命令：

```shell
git clone -b develop https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

## 模型

本项目的默认配置重现原始论文中的 CPS.resnet50.deeplabv3+(1/2 Cityscapes) 配置，其中使用50%的带标注样本，复现模型的 mIoU 为 **78.39%**。本项目复现结果与原论文结果对比如下表所示：：

| CPS.resnet50.deeplabv3+(1/2 Cityscapes) | mIOU |
| --- | --- |
| original paper | 78.77% |
| reproduced | 78.39% |

请在[此链接](https://paddleseg.bj.bcebos.com/dygraph/cross_pseudo_supervision/cityscapes/deeplabv3p_resnet50_cityscapes0.5.pdparams)下载预训练权重。

## 数据准备

使用CPS源代码所提供的Cityscapes数据集，通过[OneDrive链接](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pku_edu_cn/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=139icd)下载`city`数据集， 并将数据集`city`放至`contrib/CrossPseudoSupervision/data`文件夹下，准备好的数据组织如下：

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

执行以下命令，进入到`CrossPseudoSupervision`文件夹下：

```shell
cd ./contrib/CrossPseudoSupervision
```

### 训练

准备好环境与数据之后，执行以下命令启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml --log_iters 10 --save_dir ./output/ --batch_size 2
```

建议使用单机多卡进行训练，执行以下命令启动四卡训练：

```shell
python -m paddle.distributed.launch --gpus="0,1,2,3" train.py --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
--log_iters 10 --save_dir $SAVE_PATH$ --batch_size 8
```

- `SAVE_PATH`: 保存权重与日志等文件的文件夹路径。

**注**：
1. 配置文件是训练1/2有标签的数据，若要调整为其他比例，修改配置文件中的`labeled_ratio`参数。当修改有标签数据的比例时，训练的epoch数需要按照下表进行调整（通过修改配置文件中的`nepochs`参数调整训练的epoch数量）：

| Ratio    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| nepochs | 128  | 137  | 160  | 240  |


### 评估

训练结束后，执行以下命令评估模型精度：

```shell
export CUDA_VISIBLE_DEVICES=0
python val.py \
       --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
       --model_path $MODEL_PATH$
```

- `MODEL_PATH`: 要加载的权重路径。

### 预测

执行以下命令，使用滑窗推理进行预测：

```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py \
       --config ./configs/deeplabv3p/deeplabv3p_resnet50_0.5cityscapes_800x800_240e.yml \
       --model_path $MODEL_PATH$ \
       --image_path $IMG_PATH$ \
       --save_dir $SAVE_PATH$ \
       --is_slide \
       --crop_size 800 800 \
       --stride 532 532
```

- `IMG_PATH`: 待预测的图片或文件夹所在的路径。

本项目提供[预训练模型](https://paddleseg.bj.bcebos.com/dygraph/cross_pseudo_supervision/cityscapes/deeplabv3p_resnet50_cityscapes0.5.pdparams)可供直接进行预测。
