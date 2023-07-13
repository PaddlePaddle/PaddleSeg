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
pip install "paddleseg>=2.5"
pip install -r requirements.txt
```

## 下载预训练模型
下载[模型库](../README_CN.md/#模型库)中的预训练模型并放置于pretrained_models目录下。这边以PP—MattingV2为例。
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

预测结果如下:
<div align="center">
<img src="https://user-images.githubusercontent.com/30919197/201861635-0d139592-7da5-44b1-9bfa-7502d9643320.png"  width = "90%"  />
</div>

**注意**： `--config`需要与`--model_path`匹配。

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
背景替换效果如下：
<div align="center">
<img src="https://user-images.githubusercontent.com/30919197/201861644-15dd5ccf-fb6e-4440-a731-8e7c1d464699.png"  width = "90%"  />
</div>

**注意：**
* `--image_path`必须是一张图片的具体路径。
* `--config`需要与`--model_path`匹配。
* `--background`可以传入背景图片路径，或选择（'r','g','b','w')中的一种，代表红，绿，蓝，白背景, 若不提供则采用绿色作为背景。


## 视频预测
运行如下命令进行视频预测，切记通过`--video_path`传入待预测视频
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/predict_video.py \
    --config configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml \
    --model_path pretrained_models/ppmattingv2-stdc1-human_512.pdparams \
    --video_path path/to/video \
    --save_dir ./output/results \
    --fg_estimate True
```
预测结果如下：

<p align="center">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1.gif"  height="200">  
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_alpha.gif"  height="200">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_fg.gif"  height="200">
</p>


## 视频背景替换
运行如下命令进行视频预测，切记通过`--video_path`传入待背景替换视频
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
背景替换效果如下：
<p align="center">
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1.gif"  height="200">  
<img src="https://paddleseg.bj.bcebos.com/matting/demo/v1_bgv1.gif"  height="200">
</p>

**注意：**
* `--background`可以传入背景图片路径，或背景视频路径，或选择（'r','g','b','w')中的一种，代表红，绿，蓝，白背景, 若不提供则采用绿色作为背景。
