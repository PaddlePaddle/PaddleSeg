English | [简体中文](README_cn.md)

# Multi-label semantic segmentation based on PaddleSeg

## 1. introduction

Multi-label semantic segmentation is an image segmentation task that aims to assign each pixel in an image to multiple categories, rather than just one category. This can better express complex information in the image, such as overlapping, occlusion, boundaries, etc. of different objects. Multi label semantic segmentation has many application scenarios, such as medical image analysis, remote sensing image interpretation, autonomous driving, and so on.

<p align="center">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/ea6bb360-75de-4e06-9910-44c7d2fdbe6c">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/e2781865-db7e-4f46-98b2-3ef731e8bef1">
<img src="https://github.com/PaddlePaddle/PaddleSeg/assets/95759947/9e587935-fd6f-459e-b798-0164eb98f44d">
</p>

+ *The above effect shows the inference results obtained from the model trained using images in the [UWMGI](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/) dataset*

## 2. Supported models and loss functions

|                                            Model                                            |           Loss           |
|:-------------------------------------------------------------------------------------------:|:------------------------:|
| DeepLabV3, DeepLabV3P, MobileSeg, <br/>PP-LiteSeg, PP-MobileSeg, UNet, <br/>Unet++, Unet+++ | BCELoss, LovaszHingeLoss |

+ *The above are the confirmed supported models and loss functions, with a larger actual support range.*

## 3. Sample Tutorial

The following will take the **[UWMGI](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)** multi-label semantic segmentation dataset and the **[PP-MobileSeg](../pp_mobileseg/README.md)** model as examples.

### 3.1 Data Preparation
In the single label semantic segmentation task, the shape of the annotated grayscale image is **(img_h, img_w)**, and the index value of the category is represented by grayscale values.

In the multi-label semantic segmentation task, the shape of the annotated grayscale image is **(img_h, num_classes x img_w)**, which means that the corresponding binary annotations of each category are sequentially concatenated in the horizontal direction.

Download the raw data compression package of the UWMGI dataset and convert it to a format supported by PaddleSeg's [Dataset](../../paddleseg/datasets/dataset.py) API using the provided script.
```shell
wget https://paddleseg.bj.bcebos.com/dataset/uw-madison-gi-tract-image-segmentation.zip
python tools/data/convert_multilabel.py \
    --dataset_type uwmgi \
    --zip_input ./uw-madison-gi-tract-image-segmentation.zip \
    --output ./data/UWMGI/ \
    --train_proportion 0.8 \
    --val_proportion 0.2
# optional
rm ./uw-madison-gi-tract-image-segmentation.zip
```

The structure of the UWMGI dataset after conversion is as follows:
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

The divided training dataset and evaluation dataset can be configured as follows:
```yaml
train_dataset:
  type: Dataset
  dataset_root: data/UWMGI
  transforms:
    - type: Resize
      target_size: [512, 512]
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
      target_size: [512, 512]
    - type: Normalize
      mean: [0.0, 0.0, 0.0]
      std: [1.0, 1.0, 1.0]
  num_classes: 3
  val_path: data/UWMGI/val.txt
  mode: val
```

We add`AddMultiLabelAuxiliaryCategory` transform for add background in the segmentation to improve segmentation performance. You can config it in the first step of transform, please refer to`configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux.yml`.

### 3.2 Training
```shell
python tools/train.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux.yml \
    --save_dir output/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux \
    --num_workers 8 \
    --do_eval \
    --use_vdl \
    --save_interval 2000 \
    --use_multilabel
```
+ *When using `--do_eval`must be added `--use_multilabel` parameter is used to adapt the evaluation in multi-label mode.*

### 3.3 Evaluation
```shell
python tools/val.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux/best_model/model.pdparams \
    --use_multilabel
```
+ *Must add `--use_multilabel` when evaluating the model to adapt the evaluation in multi-label mode.*

### 3.4 Inference
```shell
python tools/predict.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_80k_withaux/best_model/model.pdparams \
    --image_path data/UWMGI/images/val/case122_day18_slice_0089.jpg \
    --use_multilabel
```
+ *When executing a prediction, it is necessary to add `--use_multilabel` parameter is used to adapt visualization in multi-label mode.*
