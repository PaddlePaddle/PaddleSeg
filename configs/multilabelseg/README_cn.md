[English](README.md) | 简体中文

# 基于 PaddleSeg 的多标签语义分割

## 1. 简介

多标签语义分割是一种图像分割任务，它的目的是将图像中的每个像素分配到多个类别中，而不是只有一个类别。这样可以更好地表达图像中的复杂信息，例如不同物体的重叠、遮挡、边界等。多标签语义分割有许多应用场景，例如医学图像分析、遥感图像解译、自动驾驶等。

<p align="center">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/ea6bb360-75de-4e06-9910-44c7d2fdbe6c">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/e2781865-db7e-4f46-98b2-3ef731e8bef1">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/9e587935-fd6f-459e-b798-0164eb98f44d">
</p>

+ *以上效果展示图基于 [UWMGI](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)数据集中的图片使用训练的模型所得到的推理结果。*

## 2. 已支持的模型和损失函数

|                                            Model                                            |           Loss           |
|:-------------------------------------------------------------------------------------------:|:------------------------:|
| DeepLabV3, DeepLabV3P, MobileSeg, <br/>PP-LiteSeg, PP-MobileSeg, UNet, <br/>Unet++, Unet+++ | BCELoss, LovaszHingeLoss |

+ *以上为确认支持的模型和损失函数，实际支持范围更大。*

## 3. 示例教程

如下将以 **[UWMGI](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)** 多标签语义分割数据集和 **[PP-MobileSeg](../pp_mobileseg/README.md)** 模型为例。

### 3.1 数据准备
在单标签多类别语义分割任务中，标注灰度图的形状为 **(img_h, img_w)**, 并以灰度值来表示类别的索引值。

在多标签语义分割任务中，标注灰度图的形状为 **(img_h, num_classes x img_w)**, 即将各个类别对应二值标注按顺序拼接在水平方向上。

下载UWMGI数据集的原始数据压缩包，并使用提供的脚本转换为PaddleSeg的[Dataset](../../paddleseg/datasets/dataset.py) API支持的格式。
```shell
wget https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/27923/3495119/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1693533809&Signature=ThCLjIYxSXfk85lCbZ5Cz2Ta4g8AjwJv0%2FgRpqpchlZLLYxk3XRnrZqappboha0moC7FuqllpwlLfCambQMbKoUjCLylVQqF0mEsn0IaJdYwprWYY%2F4FJDT2lG0HdQfAxJxlUPonXeZyZ4pZjOrrVEMprxuiIcM2kpGk35h7ry5ajkmdQbYmNQHFAJK2iO%2F4a8%2F543zhZRWsZZVbQJHid%2BjfO6ilLWiAGnMFpx4Sh2B01TUde9hBCwpxgJv55Gs0a4Z1KNsBRly6uqwgZFYfUBAejySx4RxFB7KEuRowDYuoaRT8NhSkzT2i7qqdZjgHxkFZJpRMUlDcf1RSJVkvEA%3D%3D&response-content-disposition=attachment%3B+filename%3Duw-madison-gi-tract-image-segmentation.zip
python tools/data/convert_multilabel.py \
    --dataset_type uwmgi \
    --zip_input ./uw-madison-gi-tract-image-segmentation.zip \
    --output ./data/UWMGI/ \
    --train_proportion 0.8 \
    --val_proportion 0.2
# 可选
rm ./uw-madison-gi-tract-image-segmentation.zip
```

转换完成后的UWMGI数据集结构如下：
```
UWMGI
    |
    |--images
    |  |--train
    |  |  |--*.jpg
    |  |  |--...
    |  |
    |  |--val
    |  |  |--*.jpg
    |  |  |--...
    |
    |--annotations
    |  |--train
    |  |  |--*.jpg
    |  |  |--...
    |  |
    |  |--val
    |  |  |--*.jpg
    |  |  |--...
    |
    |--train.txt
    |
    |--val.txt
```

划分好的训练数据集和评估数据集可按如下方式进行配置：
```yaml
train_dataset:
  type: Dataset
  dataset_root: data/UWMGI
  transforms:
    - type: Resize
      target_size: [256, 256]
    - type: RandomHorizontalFlip
    - type: RandomVerticalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize
      mean: [0.0, 0.0, 0.0]
      std: [1.0, 1.0, 1.0]
  num_classes: 3
  train_path: data/UWMGI/train.txt
  mode: train

val_dataset:
  type: Dataset
  dataset_root: data/UWMGI
  transforms:
    - type: Resize
      target_size: [256, 256]
    - type: Normalize
      mean: [0.0, 0.0, 0.0]
      std: [1.0, 1.0, 1.0]
  num_classes: 3
  val_path: data/UWMGI/val.txt
  mode: val
```

### 3.2 训练模型
```shell
python tools/train.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_160k.yml \
    --save_dir output/pp_mobileseg_tiny_uwmgi_256x256_160k \
    --num_workers 8 \
    --do_eval \
    --use_vdl \
    --save_interval 2000 \
    --use_multilabel
```
+ *当使用`--do_eval`必须添加`--use_multilabel`参数来适配多标签模式下的评估。*

### 3.3 评估模型
```shell
python tools/val.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_160k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_160k/best_model/model.pdparams \
    --use_multilabel
```
+ *评估模型时必须添加`--use_multilabel`参数来适配多标签模式下的评估。*

### 3.4 执行预测
```shell
python tools/predict.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_160k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_160k/best_model/model.pdparams \
    --image_path data/UWMGI/images/val/case122_day18_slice_0089.jpg \
    --use_multilabel
```
+ *执行预测时必须添加`--use_multilabel`参数来适配多标签模式下的可视化。*