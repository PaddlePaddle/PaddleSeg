English | [简体中文](README_cn.md)

# Multi-label semantic segmentation based on PaddleSeg

## 1. introduction

Multi-label semantic segmentation is an image segmentation task that aims to assign each pixel in an image to multiple categories, rather than just one category. This can better express complex information in the image, such as overlapping, occlusion, boundaries, etc. of different objects. Multi label semantic segmentation has many application scenarios, such as medical image analysis, remote sensing image interpretation, autonomous driving, and so on.

<p align="center">
<img src="https://github.com/MINGtoMING/cache_ppseg_multilabelseg_readme_imgs/tree/main/assets/case15_day0_slice_0065.jpg">
<img src="https://github.com/MINGtoMING/cache_ppseg_multilabelseg_readme_imgs/tree/main/assets/case122_day18_slice_0092.jpg">
<img src="https://github.com/MINGtoMING/cache_ppseg_multilabelseg_readme_imgs/tree/main/assets/case130_day20_slice_0072.jpg">
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
wget https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/27923/3495119/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1693533809&Signature=ThCLjIYxSXfk85lCbZ5Cz2Ta4g8AjwJv0%2FgRpqpchlZLLYxk3XRnrZqappboha0moC7FuqllpwlLfCambQMbKoUjCLylVQqF0mEsn0IaJdYwprWYY%2F4FJDT2lG0HdQfAxJxlUPonXeZyZ4pZjOrrVEMprxuiIcM2kpGk35h7ry5ajkmdQbYmNQHFAJK2iO%2F4a8%2F543zhZRWsZZVbQJHid%2BjfO6ilLWiAGnMFpx4Sh2B01TUde9hBCwpxgJv55Gs0a4Z1KNsBRly6uqwgZFYfUBAejySx4RxFB7KEuRowDYuoaRT8NhSkzT2i7qqdZjgHxkFZJpRMUlDcf1RSJVkvEA%3D%3D&response-content-disposition=attachment%3B+filename%3Duw-madison-gi-tract-image-segmentation.zip
python tools/data/convert_uwmgi.py \
    ./uw-madison-gi-tract-image-segmentation.zip \
    ./data/UWMGI/ \
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
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [256, 256]
    - type: RandomHorizontalFlip
    - type: RandomVerticalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
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
  num_classes: 3
  val_path: data/UWMGI/val.txt
  mode: val
```

### 3.2 Training
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
+ *When using `--do_eval`must be added `--use_multilabel` parameter is used to adapt the evaluation in multi-label mode.*

### 3.3 Evaluation
```shell
python tools/val.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_160k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_160k/best_model/model.pdparams \
    --use_multilabel
```
+ *Must add `--use_multilabel` when evaluating the model to adapt the evaluation in multi-label mode.*

### 3.4 Inference
```shell
python tools/predict.py \
    --config configs/multilabelseg/pp_mobileseg_tiny_uwmgi_256x256_160k.yml \
    --model_path output/pp_mobileseg_tiny_uwmgi_256x256_160k/best_model/model.pdparams \
    --image_path data/UWMGI/images/val/case122_day18_slice_0089.jpg \
    --use_multilabel
```
+ *When executing a prediction, it is necessary to add `--use_multilabel` parameter is used to adapt visualization in multi-label mode.*