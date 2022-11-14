# 快速体验

## 环境配置

#### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.2

* Python >= 3.7+

由于图像抠图模型计算开销大，推荐在GPU版本的PaddlePaddle下使用。
推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

#### 2. 下载PaddleSeg仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

#### 3. 安装

```shell
cd PaddleSeg/Matting
pip install -r requirements.txt
```

## 下载预训练模型
```shell
mkdir pretrained_models && cd pretrained_models
wget https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams
cd ..
```

## 预测
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --image_path demo/human.jpg \
    --save_dir ./output/results \
    --fg_estimate True
```

## 背景替换
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

`--background`可以传入背景图片路劲，或选择（'r','g','b','w')中的一种，代表红，绿，蓝，白背景, 若不提供则采用绿色作为背景。

**注意：** `--image_path`必须是一张图片的具体路径。
