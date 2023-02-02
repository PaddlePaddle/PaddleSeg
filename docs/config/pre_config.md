English | [简体中文 ](pre_config_cn.md)
# Config Preparation

The config file contains the information of train dataset, val dataset, optimizer, loss and model in PaddleSeg.
All config files of SOTA models are saved in `PaddleSeg/configs`.
Based on these config files, we can modify the content at will and then conduct model training.

The config file of `PaddleSeg/configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml` are as following.

## Explain Details

PaddleSeg employes the config file to build dataset, optimizer, model, etc, and then it conducts model training, evaluation and exporting.

Hyperparameters have batch_size and iters.

In each config module, `type` is the class name of corresponding component, and other values are the input params of `__init__` function.

For dataset config module,  the supported classes in `PaddleSeg/paddleseg/datasets` are registered by `@manager.DATASETS.add_component`.

For data transforms config module, the supported classes in `PaddleSeg/paddleseg/transforms/transforms.py` are registered by `@manager.TRANSFORMS.add_component`.

For optimizer config module, it supports all optimizer provided by PaddlePaddle. Please refer to the [document](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#api).

For lr_scheduler config module, it supports all lr_scheduler provided by PaddlePaddle. Please refer to the [document](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#about-lr).

For loss config module, `types` containes several loss name, `coef` defines the weights of each loss. The number of losses and weights must be equal. If all losses are the same, we can only add one loss name. All supported classes in `PaddleSeg/paddleseg/models/losses/` are registered by `@manager.LOSSES.add_component`.

For model config module, the supported classes in `PaddleSeg/paddleseg/models/` are registered by `@manager.MODELS.add_component`, and the supported backbone in `PaddleSeg/paddleseg/models/backbones` are registered by `@manager.BACKBONES.add_component`.


## Config File Demo

```
batch_size: 4  # batch size on single GPU
iters: 1000  

train_dataset:
  type: Dataset
  dataset_root: data/optic_disc_seg
  train_path: data/optic_disc_seg/train_list.txt
  num_classes: 2  # background is also a class
  mode: train
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize

val_dataset:  
  type: Dataset
  dataset_root: data/optic_disc_seg
  val_path: data/optic_disc_seg/val_list.txt
  num_classes: 2
  mode: val
  transforms:
    - type: Normalize

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1, 1, 1] # total_loss = coef_1 * loss_1 + .... + coef_n * loss_n

model:
  type: PPLiteSeg  
  backbone:  
    type: STDC2
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz

```

## Others

Note that:
* In the data transforms of train and val dataset, PaddleSeg will add read image operation in the beginning, add HWC->CHW transform operation in the end.
* For the config files in `PaddleSeg/configs/quick_start`, the learning_rate is corresponding to single GPU training. For other config files, the learning_rate is corresponding to 4 GPU training.

Besides, one config file can include another config file. For example, the right `deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml` uses `_base_` to include the left `../_base_/cityscapes.yml`.
If config value `X` in both config files (`A` includes `B`), the `X` value in `B` will be hidden.

![](./images/fig3.png)
