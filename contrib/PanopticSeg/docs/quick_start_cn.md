[English](quick_start_en.md) | 简体中文

# 快速开始

## 1 安装

请参考[此文档](./full_features_cn.md#1-安装).

## 2 使用预训练模型进行预测

### 2.1 下载预训练模型权重与示例数据

首先，请在[此链接](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/cityscapes/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k_lr0000/model.pdparams)与[此链接](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/tutorials/demo/demo.png)分别下载预训练模型权重（`model.pdparams`）与示例图像（`demo.png`）。将下载的两个文件放置于本项目的根目录。

### 2.2 执行预测

运行如下命令：

```shell
mkdir -p vis
python tools/predict.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k_lr00005.yml \
    --model_path model.pdparams \
    --image_path demo.png \
    --save_dir vis
```

需要说明的是，这里使用了动态图格式的模型进行推理。在部署阶段，通常可以考虑采用效率更高的静态图格式模型，详情请参见[部署文档](full_features_cn.md#5-模型部署)。

### 2.3 观察可视化结果

对于每幅图像，给出三个可视化结果。所有可视化结果均存储在 `vis` 目录中。

名称为 `demo_sem.png` 的图像从语义分割的角度可视化预测结果：

<img src="https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/tutorials/demo/demo_sem.png" height="300">

名称为 `demo_ins.png` 的图像从实例分割的角度可视化预测结果：

<img src="https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/tutorials/demo/demo_ins.png" height="300">

名称为 `demo_pan.png` 的图像从全景分割的角度可视化预测结果：

<img src="https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/tutorials/demo/demo_pan.png" height="300">

关于对可视化结果的具体描述可参考[此处](full_features_cn.md#43-获取可视化结果)。

## 3 模型训练与评估

### 3.1 准备数据集

本工具箱预置一系列脚本，可对部分公开的全景分割数据集进行预处理。请参考[此文档](../tools/data/README.md)了解更多细节。

### 3.2 训练模型

首先确认需要使用的配置文件，如 `configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k_lr00005.yml`。所有预置的配置文件均存储在 `configs`。接着，执行如下命令：

```shell
python tools/train.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k_lr00005.yml \
    --do_eval \
    --save_dir output
```

**请注意，某些模型可能包含『前置条件』**，例如，在执行模型训练、评估等步骤前可能需要编译外部 C++/CUDA 算子。请在 `config` 的相关文档中阅读更多细节。

### 3.3 评估模型精度指标

在训练过程中或训练完成后，在训练脚本的 `--save_dir` 选项指定的目录（默认为 `output`）中将存储模型权重等训练结果。可通过如下指令对验证集上 [PQ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) 指标最高的模型权重（也就是 `output/best_model`）进行精度评估：

```shell
python tools/val.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k_lr00005.yml \
    --model_path output/best_model/model.pdparams \
    --eval_sem \
    --eval_ins
```

默认只计算全景分割指标（如 PQ），指定 `--eval_sem` 与 `--eval_ins` 将分别计算语义分割指标（如 mIoU）与实例分割指标（如 mAP）。
