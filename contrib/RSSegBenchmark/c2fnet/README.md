# C2FNet: A Coarse-to-fine Segmentation Model for Segmentation on Remote Sensing Images

Benefiting from the [remote sensing segmentation benchmark](../README.md), a coarse-to-fine segmentation model for segmentation on remote sensing images
(C2FNet) is proposed to improve the segmentation performance of the baseline models on small objects. This repository contains the source-code and models for C2FNet.



## Requirements
```
Python: 3.7  
PaddlePaddle: 2.3.2
PaddleSeg: 2.6
```


## Training

a. Enter C2FNet directory when under the `RSSegBenchmark` directory.

```shell
cd c2fnet
```

b. Train a coarse model on [Remote Sensing Benchmark](../README.md) or use our provied [coarse models](#results).

c. Train the fine model.

```shell
# Only One GPU
export CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/c2fnet/c2fnet_{coarse_model_name}_isaid_512x512.yml \
       --coarse_model {YOUR COARSE_MODEL PATH} \
       --do_eval \
       --save_interval 8000 \
       --save_dir {OUTPUT PATH}
```

```shell
#Multiple GPUs
export CUDA_VISIBLE_DEVICES= 0,1,2,3
python -m paddle.distributed.launch train.py \
       --config configs/c2fnet/c2fnet_{coarse_model_name}_isaid_512x512.yml \
       --coarse_model {YOUR COARSE_MODEL PATH} \
       --do_eval \
       --save_interval 8000 \
       --save_dir {OUTPUT PATH}
```

## Evaluating

a. Enter C2FNet directory when under the `RSSegBenchmark` directory.

```shell
cd c2fnet
```

b. Evaluate the best fine model.
```python
python val.py \
       --config configs/c2fnet/c2fnet_{coarse_model_name}_isaid_512x512.yml \
       --coarse_model {YOUR COARSE_MODEL PATH} \
       --model_path {YOUR BEST FINE_MODEL PATH}
```

## Predicting

a. Predict with your fine model.
```python
python predict.py \
       --config configs/c2fnet/c2fnet_{coarse_model_name}_isaid_512x512.yml \
       --coarse_model {YOUR COARSE_MODEL PATH} \
       --model_path {YOUR BEST FINE_MODEL PATH} \
       --image_path {YOUR IMAGE PATH} \
       --save_dir {OUTPUT PATH}
```

## Results

| Model | Backbone | Resolution | Ship | Large_Vehicle | Small_Vehicle | Helicopter | Swimming_Pool |Plane| Harbor | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FCN | HRNet_W18 | 512x512 | 69.04 | 62.61 | 48.75 | 23.14 | 44.99 | 83.35 | 58.61 | [cfg](../configs/fcn/fcn_hrnet_w18_isaid_512_512.yml) \| [coarse_model](fcn_hrnetw18_isaid) |
|FCN_C2FNet | - | 512x512 | 69.51 | 63.60 | 51.58 | 24.47 | 46.19 | 84.04 | 60.55 | [cfg](./configs/c2fnet/c2fnet_fcn_isaid_512x512.yml) \| [model](c2fnet_fcn) |
| PSPNet | ResNet50_vd | 512x512 | 66.74 | 62.18 | 46.35 | 32.59 | 47.17 | 81.87 | 54.72 | [cfg](../configs/pspnet/pspnet_resnet50_vd_isaid_512_512.yml) \| [coarse_model](pspnet_resnet50vd_isaid) |
| PSPNet_C2FNet | - | 512x512 | 67.92 | 63.94 | 50.63 | 33.30 | 48.85 | 83.72 | 56.93 | [cfg](./configs/c2fnet/c2fnet_pspnet_isaid_512x512.yml) \| [model](c2fnet_pspnet) |
| DeeplabV3+ | ResNet50_vd | 512x512 | 64.49 | 61.68 | 45.77 | 33.35 | 49.58 | 81.65 | 53.41 | [cfg](../configs/deeplabv3%2B/deeplabv3%2B_resnet50_vd_isaid_512_512.yml) \| [coarse_model](deeplabv3+_resnet50vd_isaid) |
| DeeplabV3+_C2FNet | - | 512x512 | 70.37 | 65.50 | 51.73 | 39.04 | 48.19 | 84.83 | 58.13 | [cfg](./configs/c2fnet/c2fnet_deeplabv3plus_isaid_512x512.yml) \| [model](c2fnet_deeplabv3+) |
| HRNet | HRNet_W48 | 512x512 | 73.80 | 66.61 | 54.27 | 38.17 | 52.19 | 85.51 | 62.25 | [cfg](../configs/hrnet/hrnet_w48_isaid_512_512.yml) \| [coarse_model](hrnetw48_isaid)|
| HRNet_C2FNet | - | 512x512 | 74.32 | 67.56 | 56.46 | 38.89 | 52.78 | 85.75 | 63.70 | [cfg](./configs/c2fnet/c2fnet_hrnet_isaid_512x512.yml) \| [model](c2fnet_hrnet)|



## Contact

wangqingzhong@baidu.com

chensilin@baidu.com
