# Quick Start

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


## Download pre-trained model
Download the pre-trained model in [Models](../README.md/#Models) to `pretrained_models`. Take PP-MattingV2 as an example.
```shell
mkdir pretrained_models && cd pretrained_models
wget https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams
cd ..
```

## Prediction
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --image_path demo/human.jpg \
    --save_dir ./output/results \
    --fg_estimate True
```

Prediction results are as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/30919197/201861635-0d139592-7da5-44b1-9bfa-7502d9643320.png"  width = "90%"  />
</div>

**Note**: `--config` needs to match `--model_path`.

## Background Replacement
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --image_path demo/human.jpg \
    --background 'g' \
    --save_dir ./output/results \
    --fg_estimate True
```
The background replacement effect is as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/30919197/201861644-15dd5ccf-fb6e-4440-a731-8e7c1d464699.png"  width = "90%"  />
</div>

**Notes:**
* `--image_path` must be the specific path of an image.
* `--config` needs to match `--model_path`.
* `--background` can be passed into the background image path, or one of ('r','g','b','w'), representing a red, green, blue, or white background, default green if not passed.


## Video Prediction

Run the following commad to predict the video, and remember to pass the video path by `--video_path`.

```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --video_path path/to/video \
    --save_dir ./output/results \
    --fg_estimate True
```
Prediction results are as follows:

<p align="center">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1.gif"  height="200">  
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_alpha.gif"  height="200">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_fg.gif"  height="200">
</p>


## Video Background Replacement
Run the following commad to replace video background, and remember to pass the video path by `--video_path`.
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --video_path path/to/video \
    --background 'g' \
    --save_dir ./output/results \
    --fg_estimate True
```
The background replacement effect is as follows:
<p align="center">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1.gif"  height="200">  
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_bgv1.gif"  height="200">
</p>

**Notes:**
* `--background` can be passed into the background image path, or background video path, or one of ('r','g','b','w'), representing a red, green, blue, or white background, default green if not passed.
