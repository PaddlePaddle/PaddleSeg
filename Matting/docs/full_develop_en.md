# Full Development

## Contents
* [Installation](#Installation)
* [Dataset preparation](#Dataset-preparation)
* [Model selection](#Model-selection)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Prediction](#Prediction)
* [Background Replacement](#Background-Replacement)
* [Export and Deployment](#Export-and-Deployment)

## Installation

#### 1. Install PaddlePaddle

Versions

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

Due to the high computational cost of model, PaddleSeg is recommended for GPU version PaddlePaddle.
CUDA 10.0 or later is recommended. See [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) for the installation tutorial.

#### 2. Download the PaddleSeg repository

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### 3. Installation

```shell
cd PaddleSeg/Matting
pip install "paddleseg>=2.5"
pip install -r requirements.txt
```


## Dataset preparation

Using MODNet's open source [PPM-100](https://github.com/ZHKKKe/PPM) dataset as our demo dataset for the tutorial.
Custom dataset refer to [dataset preparation](data_prepare_en.md).

Download the prepared PPM-100 dataset.
```shell
mkdir data && cd data
wget https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip
unzip PPM-100.zip
cd ..
```

The dataset structure is as follows.

```
PPM-100/
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha
|
|--train.txt
|
|--val.txt
```

**Note** : This dataset is only used as a tutorial demonstration and cannot be trained to produce a convergent model.

## Model selection

The Matting project supports configurable direct drive, with model config files placed in [configs](../configs/) directory.
You can select a config file based on the actual situation to perform training, prediction et al.
The trimap-based methods (DIM) do not support video processing.

This tutorial uses [configs/quick_start/ppmattingv2-stdc1-human_512.yml](../configs/quick_start/ppmattingv2-stdc1-human_512.yml) for teaching demonstrations.

## Training

```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
       --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --num_workers 5 \
       --save_dir output
```

Using `--do_eval` will affect training speed and increase memory consumption, turning on and off according to needs.
If opening the `--do_eval`, the historical best model will be saved to '{save_dir}/best_model' according to SAD. At the same time, 'best_sad.txt' will be generated in this directory to record the information of metrics and iter at this time.

`--num_workers` Read data in multi-process mode. Speed up data preprocessing.

Run the following command to view more parameters.
```shell
python tools/train.py --help
```
If you want to use multiple GPUs，please use `python -m paddle.distributed.launch` to run.

## Finetune
If you want to finetune from a pretrained model, you can set the `model.pretrained` field in config file, whose content is the URL or filepath of the pretrained model weights.Here we use the official PP-MattingV2 pretrained model for finetuning as an example.

First, download the pretrained model in [Models](../README.md/#Models) to `pretrained_models`.
```shell
mkdir pretrained_models && cd pretrained_models
wget https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams
cd ..
```
Then modify the `train_dataset.dataset_root`, `val_dataset.dataset_root`, `model.pretrained` fields in the config file, meanwhile the lr is recommended to be reduced, and you can leave the rest of the config file unchanged.
```yaml
train_dataset:
  type: MattingDataset
  dataset_root: path/to/your/dataset # Path to your own dataset
  mode: train

val_dataset:
  type: MattingDataset
  dataset_root: path/to/your/dataset # Path to your own dataset
  mode: val

model:
  type: PPMattingV2
  backbone:
    type: STDC1
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz
  decoder_channels: [128, 96, 64, 32, 16]
  head_channel: 8
  dpp_output_channel: 256
  dpp_merge_type: add
  pretrained: pretrained_models/ppmattingv2-stdc1-human_512.pdparams # The pretrained model file just downloaded
lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.001  # lr is recommended to be reduced
  end_lr: 0
  power: 0.9
  warmup_iters: 1000
  warmup_start_lr: 1.0e-5
```
Finally, you can finetune the model with your dataset following the instructions in `Training`.

## Evaluation
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/val.py \
       --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
       --model_path output/best_model/model.pdparams \
       --save_dir ./output/results \
       --save_results
```
`--save_result` The prediction results will be saved if turn on. If it is off, it will speed up the evaluation.

You can directly download the provided model for evaluation.

Run the following command to view more parameters.
```shell
python tools/val.py --help
```

## Prediction
### Image Prediction
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --image_path data/PPM-100/val/fg/ \
    --save_dir ./output/results \
    --fg_estimate True
```
If the model requires trimap information, pass the trimap path through '--trimap_path'.

`--fg_estimate False` can turn off foreground estimation, which improves prediction speed but reduces image quality.

You can directly download the provided model for evaluation.

Run the following command to view more parameters.
```shell
python tools/predict.py --help
```

### Video Prediction
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --video_path path/to/video \
    --save_dir ./output/results \
    --fg_estimate True
```


## Background Replacement
### Image Background Replacement
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --image_path path/to/your/image \
    --background path/to/your/background/image \
    --save_dir ./output/results \
    --fg_estimate True
```
If the model requires trimap information, pass the trimap path through `--trimap_path`.

`--background` can pass a path of brackground image or select one of ('r', 'g', 'b', 'w') which represent red, green, blue and white. If it is not specified, a green background is used.

`--fg_Estimate False` can turn off foreground estimation, which improves prediction speed but reduces image quality.

**note：** `--image_path` must be a image path。

You can directly download the provided model for background replacement.

Run the following command to view more parameters.
```shell
python tools/bg_replace.py --help
```

### Video Background Replacement
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --video_path path/to/video \
    --background 'g' \
    --save_dir ./output/results \
    --fg_estimate True
```

## Export and Deployment
### Model Export
```shell
python tools/export.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path output/best_model/model.pdparams \
    --save_dir output/export \
    --input_shape 1 3 512 512
```
If the model requires trimap information such as DIM, `--trimap` is need.

Run the following command to view more parameters.
```shell
python tools/export.py --help
```

### Deployment
```shell
python deploy/python/infer.py \
    --config output/export/deploy.yaml \
    --image_path data/PPM-100/val/fg/ \
    --save_dir output/results \
    --fg_estimate True
```
If the model requires trimap information, pass the trimap path through '--trimap_path'.

`--fg_Estimate False` can turn off foreground estimation, which improves prediction speed but reduces image quality.

`--video_path` can pass a video path to have a video matting.

Run the following command to view more parameters.
```shell
python deploy/python/infer.py --help
```
