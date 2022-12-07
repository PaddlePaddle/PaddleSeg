# Remote Sensing Benchmark

## 1 Introduction

To understand more topographic information of objects, semantic segmentation is widely applied to remote sensing images and contributes to land cover mapping, disaster prediction, urban planning and so on. PaddleSeg releases **A Benchmark for Semantic Segmentation on Remote Sensing Images** to provide standard configuration and several comparable baseline models. Moreover, we design **A Coarse-to-Fine Model (C2FNet)**, which optimizes the baseline methods to achieve accurate segmentation of small objects.

## 2 Model Performance

### 2.1 Baseline Models

The Benchmark implements some baseline models on iSAID, ISPRS Potsdam and ISPRS Vaihingen datasets. The results are as follows.

### iSAID
| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DANet | 512x512 | ResNet50 | 80000 | 37.30 | [cfg](./configs/danet/danet_resnet50_isaid_512_512.yml) \| [model](danet_resnet) |
| DANet | 512x512 | ResNet50_vd | 80000 | 64.56 | [cfg](./configs/danet/danet_resnet50_vd_isaid_512_512.yml) \| [model](danet_resnetvd) |
| DeeplabV3+ | 512x512 | ResNet50 | 80000 | 62.59 | [cfg](./configs/deeplabv3%2B/deeplabv3%2B_resnet50_isaid_512_512.yml) \| [model](deeplab_resnet) |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 65.46 | [cfg](./configs/deeplabv3%2B/deeplabv3%2B_resnet50_vd_isaid_512_512.yml) \| [model](deeplab_resnetvd) |
| FCN | 512x512 | ResNet50 | 80000 | 52.12 | [cfg](./configs/fcn/fcn_resnet50_isaid_512_512.yml) \| [model](fcn_resnet) |
| FCN | 512x512 | HRNet_W18 | 80000 | 64.73 | [cfg](./configs/fcn/fcn_hrnet_w18_isaid_512_512.yml) \| [model](fcn_hrnet_w18) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 67.31 | [cfg](./configs/hrnet/hrnet_w48_isaid_512_512.yml) \| [model](hrnetw48) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 63.36 | [cfg](./configs/pspnet/pspnet_resnet50_vd_isaid_512_512.yml) \| [model](pspnet_resnet50vd) |
| UperNet | 512x512 | ResNet50 | 80000 | 64.10 | [cfg](./configs/upernet/upernet_resnet50_isaid_512_512.yml) \| [model](upernet_resnet50) |

### ISPRS Potsdam

| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 77.93 | [cfg](./configs/deeplabv3%2B/deeplabv3%2B_resnet50_vd_potsdam.yml) \| [model](deeplabv3_potsdam) |
| PSPNet | 512x512 | ResNet50_vd | 80000 | 77.69 | [cfg](./configs/pspnet/pspnet_resnet50_vd_potsdam.yml) \| [model](pspnet_potsdam) |
| FCN | 512x512 | HRNet_W18 | 80000 | 78.13 | [cfg](./configs/fcn/fcn_hrnet_w18_potsdam.yml) \| [model](fcn_hrnet_potsdam) |
| HRNet | 512x512 | HRNet_W48 | 80000 | 78.84 | [cfg](./configs/hrnet/hrnet_w48_potsdam.yml) \| [model](hrnet_w48_potsdam) |
| UperNet | 512x512 | ResNet50 | 80000 | 77.59 | [cfg](./configs/upernet/upernet_resnet50_potsdam.yml) \| [model](upernet_potsdam) |



### ISPRS Vaihingen

| Model | Resolution | Backbone | Iters | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| DeeplabV3+ | 512x512 | ResNet50_vd | 80000 | 74.08 | [cfg](./configs/deeplabv3%2B/deeplabv3%2B_resnet50_vd_vaihingen.yml) \| [model](deeplab_vaihingen) |
| FCN | 512x512 | HRNet_W18 | 80000 | 73.25 | [cfg](./configs/fcn/fcn_hrnet_w18_vaihingen.yml) \| [model](fcn_hrnet_vaihingen)|
| HRNet | 512x512 | HRNet_W48 | 80000 | 74.98 | [cfg](./configs/hrnet/hrnet_w48_vaihingen.yml) \| [model](hrnetw48_vaihingen) |
| UperNet | 512x512 | ResNet50_vd | 80000 | 74.31 | [cfg](./configs/upernet/upernet_resnet50_vd_vaihingen.yml) \| [model](upernet_resnet50vd_vaihingen)|}

## 2.2 Self-supervised Pretrained Models

We investigate the generalization ability of the self-supervised learning in remote sensing images based on [PASSL](https://github.com/PaddlePaddle/PASSL). Some valuable experimental results and self-supervised pre-training models are provided in our benchmark to facilitate further research on self-supervised learning in the remote sensing images.

### ImageNet Pretrianed Models by Self-supervised Learning.

| Dataset | Segmentor | SSL | Backbone | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | DeeplabV3+ | PixelPro | ResNet50 | 62.22 | [cfg](./configs/ssl/deeplabv3%2B_pixpro_imgnet_resnet50_isaid_512_512.yml) \| [model](deeplabv3+_pixpro_imgnet) |
| iSAID | DeeplabV3+ | DenseCL | ResNet50 | 56.94 | [cfg](./configs/ssl/deeplabv3%2B_densecl_imgnet_resnet50_isaid_512_512.yml) \| [model](deeplabv3+_densecl_imgnet) |
| iSAID | DeeplabV3+ | MoCoBYOL | ResNet50 | 57.96 | [cfg](./configs/ssl/deeplabv3%2B_mocobyol_imgnet_resnet50_isaid_512_512.yml) \| [model](deeplabv3+_mocobyol_imgnet) |
| iSAID | FCN | PixelPro | ResNet50 | 51.30 | [cfg](./configs/ssl/fcn_pixpro_imgnet_resnet50_isaid_512_512.yml) \| [model](fcn_pixpro_imgnet) |
| iSAID | OCRNet | PixelPro | ResNet50 | 41.95 | [cfg](./configs/ssl/ocrnet_pixpro_imgnet_resnet50_isaid_512_512.yml) \| [model](ocrnet_pixpro_imgnet) |
| iSAID | PSPNet | PixelPro | ResNet50 | 50.23 | [cfg](./configs/ssl/pspnet_pixpro_imgnet_resnet50_isaid_512_512.yml) \| [model](pspnet_pixpro_imgnet) |
| iSAID | UperNet | PixelPro | ResNet50 | 64.36 | [cfg](./configs/ssl/upernet_pixpro_imgnet_resnet50_isaid_512_512.yml) \| [model](upernet_pixpro_imgnet) |
| iSAID | UperNet | DenseCL | ResNet50 | 54.22 | [cfg](./configs/ssl/upernet_densecl_imgnet_resnet50_isaid_512_512.yml) \| [model](upernet_densecl_imgnet) |
| iSAID | UperNet | MoCoBYOL | ResNet50 | 64.36 | [cfg](./configs/ssl/upernet_mocobyol_imgnet_resnet50_isaid_512_512.yml) \| [model](upernet_byol_imgnet) |
| iSAID | UperNet | SimSiam | ResNet50 | 50.70 | [cfg](./configs/ssl/upernet_simsiam_imgnet_resnet50_isaid_512_512.yml) \| [model](upernet_simsiam_imgnet) |
| iSAID | UperNet | SwAV | ResNet50 | 63.42 | [cfg](./configs/ssl/upernet_swav_imgnet_resnet50_isaid_512_512.yml) \| [model](upernet_swav_imgnet) |
| Potsdam | UperNet | PixelPro | ResNet50 | 77.40 | [cfg](./configs/ssl/upernet_pixpro_imgnet_resnet50_potsdam.yml) \| [model](upernet_pixpro_imgnet_potsdam) |

*Note: All the pretrained backbone networks are download from [PASSL](https://github.com/PaddlePaddle/PASSL)*

### Remote Sensing Images Pretrianed Models by Self-supervised Learning.

We perform self-supervised learning  on two remote sensing datasets, [Million-AID](https://paperswithcode.com/dataset/million-aid) and [DOTA2.0](https://captain-whu.github.io/DOTA/dataset.html). We cropped the images of different resolutions in both datasets uniformly to 512x512. The Million-AID dataset is cropped to contain **2.5 million** image patches and the DOTA dataset contained **1.7 million** image patches.

| Dataset | Segmentor | SSL | Backbone | mIoU(%) | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- |
| iSAID | UperNet | PixelPro Million-AID | ResNet50 | 58.24 | [cfg](./configs/ssl/upernet_pixpro_millionaid_resnet50_potsdam.yml) \| [model](upernet_pixpro_aid) |
| iSAID | UperNet | DenseCL Million-AID | ResNet50 | 61.67 | [cfg](./configs/ssl/upernet_densecl_millionaid_resnet50_isaid_512_512.yml) \| [model](upernet_densecl_aid) |
| iSAID | UperNet | MoCoV2 Million-AID | ResNet50 | 55.62 | [cfg](./configs/ssl/upernet_mocov2_millionaid_resnet50_isaid_512_512.yml) \| [model](upernet_mocov2_aid) |
| Potsdam | UperNet | PixelPro Million-AID | ResNet50 | 75.68 | [cfg](./configs/ssl/upernet_pixpro_millionaid_resnet50_potsdam.yml) \| [model](upernet_pixpro_aid_potsdam) |
| iSAID | UperNet | PixelPro DOTA | ResNet50 | 59.50 |[cfg](./configs/ssl/upernet_pixpro_dota_resnet50_isaid_512_512.yml) \| [model](upernet_pixpro_dota) |
| iSAID | FCN | PixelPro DOTA | ResNet50 | 41.26 | [cfg](./configs/ssl/fcn_pixpro_dota_resnet50_isaid_512_512.yml) \| [model](fcn_pixpro_dota)|

## 3 Installation

### Requirements

* Python: 3.7+  
* PaddlePaddle: 2.3.2
* PaddleSeg: 2.6


### Install
a. Create a conda environment and activate it.
```
conda create -n rsseg python=3.7
conda activate rsseg
```

b. Install PadddlePaddle [office website](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) (the version >= 2.3).

c. Download the repository.
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

d. Install dependencies
```shell
cd PaddleSeg
pip install -r requirements.txt
```

e. Build

```shell
cd PaddleSeg
python setup.py install
```

f. Go to the benchmark directory

```shell
cd PaddleSeg/contrib/RSSegBenchmark
```


## 4 Dataset Preparation

a. Download the dataset

+ [iSAID](https://captain-whu.github.io/iSAID)
+ [ISPR Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
+ [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

b. Process the dataset

For iSAID

```python
python data/prepare_isaid.py {PATH OF ISAID}
```

For ISPRS Potsdam

```python
python data/prepare_potsdam.py {PATH OF POTSDAM}
```

For ISPRS Vaihingen

```python
python data/prepare_vaihingen.py {PATH OF VAIHINGEN}
```

*Note: `train.txt`、`val.txt`、`label.txt` are generated using reference [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/data/marker/marker_cn.md)*


## 5 Model Training/Evaluating

### Training
Training with one GPU

```shell
export CUDA_VISIBLE_DEVICES=0
python ../../train.py \
       --config configs/{YOUR CONFIG FILE} \
       --do_eval \
       --save_interval 8000 \
       --save_dir {OUTPUT PATH}
```
Training with multi-gpus

```
export CUDA_VISIBLE_DEVICES= 0,1,2,3
python -m paddle.distributed.launch ../../train.py \
      --config configs/{YOUR CONFIG FILE} \
      --do_eval \
      --save_interval 8000 \
      --save_dir {OUTPUT PATH}
```

*Note: other training details can be seen in [here](../../docs/train/train.md)*


### Evaluating

Evaluation on the best model
```
python ../../val.py \
      --config configs/{YOUR CONFIG FILE} \
      --model_path {YOUR BEST MODEL PATH}
```

*Note: more evaluation details in [here](../../docs/evaluation/evaluate/evaluate.md)*

Prediction


```bash
python ../../predict.py \
       --config configs/{YOUR CONFIG FILE} \
       --model_path {YOUR BEST MODEL PATH}
       --image_path {IMAGE PATH}\
       --save_dir {OUTPUT DIR}}
```

*Note: more details in [here](../../docs/evaluation/evaluate/evaluate.md)*


## Contact

wangqingzhong@baidu.com

chensilin@baidu.com
