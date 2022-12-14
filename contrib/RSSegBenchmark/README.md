English | [简体中文](README_CN.md)

# A Benchmark for Segmentation on Remote Sensing Images

## 1 Introduction

To understand more topographic information of objects, semantic segmentation is widely applied to remote sensing images and contributes to land cover mapping, disaster prediction, urban planning and so on.
This project uses PaddleSeg for semantic segmentation on remote sensing images. The main contributions are summarized as follows:

* **A Benchmark for Semantic Segmentation on Remote Sensing Images** is provided with standard configuration and several comparable baseline models.
* **Self-supervised learning** based pre-trained models for semantic segmentation on remote sensing images are provided to facilitate further research.
* **A Coarse-to-Fine Model (C2FNet)** is proposed on the above benchmark to optimizes these baseline methods to achieve accurate segmentation of small objects. Check [here](./c2fnet/README.md) for details.

## 2 Model Performance

### 2.1 Baseline Models

The Benchmark implements some baseline models on [iSAID](https://captain-whu.github.io/iSAID), [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) and [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) datasets. The results are as follows.

#### 2.1.1 iSAID
| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DANet | 512x512 | ResNet50 | 80000 | 37.30 | [cfg](./configs/danet/danet_resnet50_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/danet_resnet50_isaid_512x512_80k/model.pdparams) |
| DANet | 512x512 | ResNet50_vd | 80000 | 64.56 | [cfg](./configs/danet/danet_resnet50_vd_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/danet_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| DeeplabV3+ | 512x512 | ResNet50 | 80000 | 62.59 | [cfg](./configs/deeplabv3p/deeplabv3p_resnet50_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/deeplabv3p_resnet50_isaid_512x512_80k/model.pdparams) |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 65.46 | [cfg](./configs/deeplabv3p/deeplabv3p_resnet50_vd_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/deeplabv3p_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 64.73 | [cfg](./configs/fcn/fcn_hrnet_w18_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/fcn_hrnet_w18_isaid_512x512_80k/model.pdparams) |
| FCN | 512x512 | ResNet50 | 80000 | 52.12 | [cfg](./configs/fcn/fcn_resnet50_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/fcn_resnet50_isaid_512x512_80k/model.pdparams) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 67.31 | [cfg](./configs/hrnet/hrnet_w48_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/hrnet_w48_isaid_512x512_80k/model.pdparams) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 63.36 | [cfg](./configs/pspnet/pspnet_resnet50_vd_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/pspnet_resnet50_vd_isaid_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50 | 80000 | 64.10 | [cfg](./configs/upernet/upernet_resnet50_isaid_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/upernet_resnet50_isaid_512x512_80k/model.pdparams) |

#### 2.1.2 ISPRS Potsdam

| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 77.93 | [cfg](./configs/deeplabv3p/deeplabv3p_resnet50_vd_potsdam_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/potsdam/deeplabv3p_resnet50_vd_potsdam_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 78.13 | [cfg](./configs/fcn/fcn_hrnet_w18_potsdam_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/potsdam/fcn_hrnet_w18_potsdam_512x512_80k/model.pdparams) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 78.84 | [cfg](./configs/hrnet/hrnet_w48_potsdam_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/potsdam/hrnet_w48_potsdam_512x512_80k/model.pdparams) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 77.69 | [cfg](./configs/pspnet/pspnet_resnet50_vd_potsdam_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/potsdam/pspnet_resnet50_vd_potsdam_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50 | 80000 | 77.59 | [cfg](./configs/upernet/upernet_resnet50_potsdam_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/potsdam/upernet_resnet50_potsdam_512x512_80k/model.pdparams) |



#### 2.1.3 ISPRS Vaihingen

| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 74.08 | [cfg](./configs/deeplabv3p/deeplabv3p_resnet50_vd_vaihingen_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/deeplabv3p_resnet50_vd_vaihingen_512x512_80k/model.pdparams) |
| FCN | 512x512 | HRNet_W18 | 80000 | 73.25 | [cfg](./configs/fcn/fcn_hrnet_w18_vaihingen_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/fcn_hrnet_w18_vaihingen_512x512_80k/model.pdparams)|
| HRNet | 512x512 | HRNet_W48 | 80000 | 74.98 | [cfg](./configs/hrnet/hrnet_w48_vaihingen_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/hrnet_w48_vaihingen_512x512_80k/model.pdparams) |
| UperNet | 512x512 | ResNet50_vd | 80000 | 74.31 | [cfg](./configs/upernet/upernet_resnet50_vd_vaihingen_512x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/vaihingen/upernet_resnet50_vd_vaihingen_512x512_80k/model.pdparams)|}

### 2.2 Self-supervised Pretrained Models

We investigate the generalization ability of the self-supervised learning in remote sensing images based on [PASSL](https://github.com/PaddlePaddle/PASSL). Some valuable experimental results and self-supervised pre-training models are provided in our benchmark to facilitate further research on self-supervised learning in the remote sensing images.

#### 2.2.1 ImageNet Pretrianed Models by Self-supervised Learning

| Dataset | Segmentor | SSL | Backbone | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | DeeplabV3+ | DenseCL | ResNet50 | 56.94 | [cfg](./configs/ssl/deeplabv3p_densecl_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_densecl_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | DeeplabV3+ | MoCoBYOL | ResNet50 | 57.96 | [cfg](./configs/ssl/deeplabv3p_mocobyol_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_mocobyol_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | DeeplabV3+ | PixelPro | ResNet50 | 62.22 | [cfg](./configs/ssl/deeplabv3p_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/deeplabv3p_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | FCN | PixelPro | ResNet50 | 51.30 | [cfg](./configs/ssl/fcn_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/fcn_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | OCRNet | PixelPro | ResNet50 | 41.95 | [cfg](./configs/ssl/ocrnet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/ocrnet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | PSPNet | PixelPro | ResNet50 | 50.23 | [cfg](./configs/ssl/pspnet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/pspnet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | DenseCL | ResNet50 | 54.22 | [cfg](./configs/ssl/upernet_densecl_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_densecl_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | MoCoBYOL | ResNet50 | 64.36 | [cfg](./configs/ssl/upernet_mocobyol_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_mocobyol_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro | ResNet50 | 64.36 | [cfg](./configs/ssl/upernet_pixpro_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | SimSiam | ResNet50 | 50.70 | [cfg](./configs/ssl/upernet_simsiam_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_simsiam_imgnet_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | SwAV | ResNet50 | 63.42 | [cfg](./configs/ssl/upernet_swav_imgnet_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_swav_imgnet_resnet50_isaid_512x512/model.pdparams) |
| Potsdam | UperNet | PixelPro | ResNet50 | 77.40 | [cfg](./configs/ssl/upernet_pixpro_imgnet_resnet50_potsdam_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_imgnet_resnet50_potsdam_512x512/model.pdparams) |

*Note: All the pretrained backbone networks were downloaded from [PASSL](https://github.com/PaddlePaddle/PASSL).*

#### 2.2.2 Remote Sensing Images Pretrianed Models by Self-supervised Learning

We perform self-supervised learning  on two remote sensing datasets, [Million-AID](https://paperswithcode.com/dataset/million-aid) and [DOTA2.0](https://captain-whu.github.io/DOTA/dataset.html). We cropped the images of different resolutions in both datasets uniformly to 512x512. The Million-AID dataset is cropped to contain **2.5 million** image patches and the DOTA dataset contained **1.7 million** image patches.

| Dataset | Segmentor | SSL | Backbone | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | FCN | PixelPro DOTA | ResNet50 | 41.26 | [cfg](./configs/ssl/fcn_pixpro_dota_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/fcn_pixpro_dota_resnet50_isaid_512x512/model.pdparams)|
| iSAID | UperNet | DenseCL Million-AID | ResNet50 | 61.67 | [cfg](./configs/ssl/upernet_densecl_millionaid_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_densecl_millionaid_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | MoCoV2 Million-AID | ResNet50 | 55.62 | [cfg](./configs/ssl/upernet_mocov2_millionaid_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_mocov2_millionaid_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro DOTA | ResNet50 | 59.50 |[cfg](./configs/ssl/upernet_pixpro_dota_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_dota_resnet50_isaid_512x512/model.pdparams) |
| iSAID | UperNet | PixelPro Million-AID | ResNet50 | 58.24 | [cfg](./configs/ssl/upernet_pixpro_millionaid_resnet50_isaid_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_millionaid_resnet50_isaid_512x512/model.pdparams) |
| Potsdam | UperNet | PixelPro Million-AID | ResNet50 | 75.68 | [cfg](./configs/ssl/upernet_pixpro_millionaid_resnet50_potsdam_512x512.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/isaid/ssl/upernet_pixpro_millionaid_resnet50_potsdam_512x512/model.pdparams) |

## 3 Installation

### 3.1 Requirements

* Python: 3.7+  
* PaddlePaddle: 2.3.2
* PaddleSeg: 2.6


### 3.2 Install
a. Create a conda environment and activate it.
```shell
conda create -n rsseg python=3.7
conda activate rsseg
```

b. Install PadddlePaddle [office website](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) (the version >= 2.3).

c. Download the repository.
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

d. Install PaddleSeg [office website](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/install.md)

f. Go to the benchmark directory.

```shell
cd PaddleSeg/contrib/RSSegBenchmark
```


## 4 Dataset Preparation

a. Download the dataset.

+ [iSAID](https://captain-whu.github.io/iSAID)
+ [ISPR Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
+ [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

b. Process the dataset.

For iSAID.

```python
python data/prepare_isaid.py {PATH OF ISAID}
```

For ISPRS Potsdam.

```python
python data/prepare_potsdam.py {PATH OF POTSDAM}
```

For ISPRS Vaihingen.

```python
python data/prepare_vaihingen.py {PATH OF VAIHINGEN}
```

*Note: `train.txt`、`val.txt`、`label.txt` are generated using reference [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/data/marker/marker_cn.md).*


## 5 Model Training/Evaluating

### 5.1 Training
Training with one GPU.

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/{YOUR CONFIG FILE} \
       --do_eval \
       --save_interval 8000 \
       --save_dir {OUTPUT PATH}
```
Training with multiple GPUs.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py \
      --config configs/{YOUR CONFIG FILE} \
      --do_eval \
      --save_interval 8000 \
      --save_dir {OUTPUT PATH}
```

*Note: For more details about training settings, please check the [PaddleSeg document](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/train/train.md).*


### 5.2 Evaluating

Get evaluation metrics of the best model.

```shell
python val.py \
      --config configs/{YOUR CONFIG FILE} \
      --model_path {YOUR BEST MODEL PATH}
```

*Note: For more details about evaluating settings, please check the [PaddleSeg document](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/evaluation/evaluate.md).*

### 5.3 Predicting

Predict and save the segmentation result of an input image with the best model.
```shell
python predict.py \
       --config configs/{YOUR CONFIG FILE} \
       --model_path {YOUR BEST MODEL PATH}
       --image_path {IMAGE PATH}\
       --save_dir {OUTPUT DIR}}
```

*Note: For more details about predicting settings, please check the [PaddleSeg document](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/predict/predict.md).*


## Contact

wangqingzhong@baidu.com

chensilin@baidu.com
