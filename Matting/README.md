English | [简体中文](README_CN.md)

# Natural Image Matting
Image Matting is the technique of extracting foreground from an image by calculating its color and transparency. It is widely used in the film industry to replace background, image composition, and visual effects. Each pixel in the image will have a value that represents its foreground transparency, called Alpha. The set of all Alpha values in an image is called Alpha Matte. The part of the image covered by the mask can be extracted to complete foreground separation.

<p align="center">
<img src="https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif" width="100%" height="100%">
</p>

# One-click experience
Try the online demo "[Non-Code Matting](http://seg.itmanbu.com/)".

<p align="center">
<img src="https://user-images.githubusercontent.com/48433081/165077834-c3191509-aeaf-45c8-b226-656174f4c152.gif" width="100%" height="100%">
</p>

## Update Notes
2022.07
[1] Release PPMatting code.
[2] Add ClosedFormMatting, KNNMatting, FastMatting, LearningBaseMatting and RandomWalksMatting traditional machine learning algorithms.
[3] Add GCA model.
[4] Improving the directory structure.
[5] Support to specify metrics for evaluation.

2022.04
[1] Add PPMatting model.
[2] Add PPHumanMatting high-resolution human matting model.
[3] Add Grad, Conn evaluation metrics.
[4] Add foreground evaluation funciton. use [ML](https://arxiv.org/pdf/2006.14970.pdf) algorithm to evaluate foreground when prediction or background replacement.
[5] Add GradientLoss and LaplacianLoss.
[6] Add RandomSharpen, RandomSharpen, RandomReJpeg, RSSN data augmentation strategies.

2021.11 Matting Project is released.
[1] Support Matting models: DIM, MODNet.
[2] Support model export and python deployment.
[3] Support background replacement function.
[4] Support human matting deployment in Android.

## Contents
- [Installation](#Installation)
- [Models](#Models)
- [Dataset Preparation](#Dataset-Preparation)
- [Training, Evaluation and Prediction](#Training-Evaluation-and-Prediction)
- [Background Replacement](#Background-Replacement)
- [Export and Deploy](#Export-and-Deploy)
- [Human Matting Deployment in Android](./deploy/human_matting_android_demo/README.md)

## Installation

#### 1. Install PaddlePaddle

Versions

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

Due to the high computational cost of model, PaddleSeg is recommended for GPU version PaddlePaddle. CUDA 10.0 or later is recommended. See [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) for the installation tutorial.

#### 2. Download the PaddleSeg repository

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### 3. Installation

```shell
cd PaddleSeg/Matting
pip install -r requirements.txt
```

## Models
A variety of human matting models are provided for you to select according the actual situation.

Model recommend:
- for accuracy: PP-Matting, using PP-Matting-512 in low resolution situation, using PP-Matting-1024 in high resolution situation.
- for speed: ModNet-MobileNetV2.
- high resolution (>2048) human matting with simple background: PP-HumanMatting.
- providing trimap：DIM-VGG16.

| Model | Params(M) | FLOPs(G) | FPS | Checkpoint | Inference Model |
| - | - | -| - | - | - |
| PP-Matting-512     | 24.5 | 91.28 | 28.9 | [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-matting-hrnet_w18-human_512.zip) |
| PP-Matting-1024    | 24.5 | 91.28 | 13.4(1024X1024) | [model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_1024.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-matting-hrnet_w18-human_1024.zip) |
| PP-HumanMatting    | 63.9 | 135.8 (2048X2048)| 32.8(2048X2048)| [model](https://paddleseg.bj.bcebos.com/matting/models/human_matting-resnet34_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/pp-humanmatting-resnet34_vd.zip) |
| ModNet-MobileNetV2 | 6.5 | 15.7 | 68.4 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-mobilenetv2.zip) |
| ModNet-ResNet50_vd | 92.2 | 151.6 | 29.0 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-resnet50_vd.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-resnet50_vd.zip) |
| ModNet-HRNet_W18   | 10.2 | 28.5 | 32.6 | [model](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/modnet-hrnet_w18.zip) |
| DIM-VGG16          | 28.4 | 175.5| 30.4 | [model](https://paddleseg.bj.bcebos.com/matting/models/dim-vgg16.pdparams) | [model inference](https://paddleseg.bj.bcebos.com/matting/models/deploy/dim-vgg16.zip) |

Note: The model default input size is (512, 512) while calcuting FLOPs and FPS and the GPU is Tesla V100 32G.

## Dataset preparation

Using MODNet's open source [PPM-100](https://github.com/ZHKKKe/PPM) dataset as our demo dataset for the tutorial.

Organize the dataset into the following structure and place the dataset under the `data` directory.

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

The image name in the fg directory must be the same as the that in the alpha directory.

The contents of train.txt and val.txt are as follows:
```
train/fg/14299313536_ea3e61076c_o.jpg
train/fg/14429083354_23c8fddff5_o.jpg
train/fg/14559969490_d33552a324_o.jpg
...
```

You can download the organized [PPM-100](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip) dataset directly for subsequent tutorials.

If the full image is composited of foreground and background like the Composition-1k dataset used in [Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf), the dataset should be organized as follows:
```
Composition-1k/
|--bg/
|
|--train/
|  |--fg/
|  |--alpha/
|
|--val/
|  |--fg/
|  |--alpha/
|  |--trimap/ (if existing)
|
|--train.txt
|
|--val.txt
```
The contents of train.txt is as follows:
```
train/fg/fg1.jpg bg/bg1.jpg
train/fg/fg2.jpg bg/bg2.jpg
train/fg/fg3.jpg bg/bg3.jpg
...
```

The contents of val.txt is as follows. If trimap does not exist in dataset, the third column is not needed and the code will generate trimap automatically.
```
val/fg/fg1.jpg bg/bg1.jpg val/trimap/trimap1.jpg
val/fg/fg2.jpg bg/bg2.jpg val/trimap/trimap2.jpg
val/fg/fg3.jpg bg/bg3.jpg val/trimap/trimap3.jpg
...
```
## Training, Evaluation and Prediction
### Training
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
       --config configs/quick_start/modnet-mobilenetv2.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --num_workers 5 \
       --save_dir output
```

**note:** Using `--do_eval` will affect training speed and increase memory consumption, turning on and off according to needs.
If opening the `--do_eval`, the historical best model will be saved to '{save_dir}/best_model' according to SAD. At the same time, 'best_sad.txt' will be generated in this directory to record the information of metrics and iter at this time.

`--num_workers` Read data in multi-process mode. Speed up data preprocessing.

Run the following command to view more parameters.
```shell
python tools/train.py --help
```
If you want to use multiple GPUs，please use `python -m paddle.distributed.launch` to run.

### Evaluation
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/val.py \
       --config configs/quick_start/modnet-mobilenetv2.yml \
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

### Prediction
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --image_path data/PPM-100/val/fg/ \
    --save_dir ./output/results \
    --fg_estimate True
```
If the model requires trimap information, pass the trimap path through '--trimap_path'.

`--fg_Estimate False` can turn off foreground estimation, which improves prediction speed but reduces image quality.

You can directly download the provided model for evaluation.

Run the following command to view more parameters.
```shell
python tools/predict.py --help
```

## Background Replacement
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/bg_replace.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
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

## Export and Deploy
### Model Export
```shell
python tools/export.py \
    --config configs/quick_start/modnet-mobilenetv2.yml \
    --model_path output/best_model/model.pdparams \
    --save_dir output/export
```
If the model requires trimap information, `--trimap` is need.

Run the following command to view more parameters.
```shell
python tools/export.py --help
```

### Deploy
```shell
python deploy/python/infer.py \
    --config output/export/deploy.yaml \
    --image_path data/PPM-100/val/fg/ \
    --save_dir output/results \
    --fg_estimate True
```
If the model requires trimap information, pass the trimap path through '--trimap_path'.

`--fg_Estimate False` can turn off foreground estimation, which improves prediction speed but reduces image quality.

Run the following command to view more parameters.
```shell
python deploy/python/infer.py --help
```
## Acknowledgement

* Thanks [Qian bin](https://github.com/qianbin1989228) for their contributons.

* Thanks for the algorithm support of [GFM](https://arxiv.org/abs/2010.16188).
