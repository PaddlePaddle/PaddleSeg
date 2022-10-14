# PSSL: Distilling ensemble of explanations for weakly-supervised pre-training of image segmentation models.

## Reference

> Xuhong Li, Haoyi Xiong, Yi Liu, Dingfu Zhou, Zeyu Chen, Yaqing Wang, and Dejing Dou. "Distilling ensemble of explanations for weakly-supervised pre-training of image segmentation models." Machine Learning (2022): 1-17. https://arxiv.org/abs/2207.03335

## Overview

In our work, we propose a method that leverages Pseudo Semantic Segmentation Labels (PSSL), to enable the end-to-end pre-training for image segmentation models based on classification datasets. PSSL was inspired by the observation that the explanation results of classification models, obtained through explanation algorithms such as CAM, SmoothGrad and LIME, would be close to the pixel clusters of visual objects. Specifically, PSSL is obtained for each image by interpreting the classification results and aggregating an ensemble of explanations queried from multiple classifiers to lower the bias caused by single models. With PSSL for every image of ImageNet, the proposed method leverages a weighted segmentation learning procedure to pre-train the segmentation network en masse.

<div align="center">
<img src="https://user-images.githubusercontent.com/13829174/177077386-0dc77e5b-2832-45ae-bfdb-37c0a5e75c19.jpg" alt="arch"  />
</div>


## Training

Our method improves the pre-training step of segmentation models. We provide the details of the pre-training here, but note that **the pre-training step can be skipped because we have provided the pre-trained models**. The PSSL dataset can be obtained by sending an email to paddleseg@baidu.com via an **official email** (not use qq, gmail, etc.) including your institution/company information and the purpose on the dataset.

Here we show the configuration files of two lightweight models, [STDC2](https://paddleseg.bj.bcebos.com/dygraph/pssl/stdc2_pssl_pretrained/model.pdparams) and [PPLite-Seg-B](https://paddleseg.bj.bcebos.com/dygraph/pssl/pp_liteseg_stdc2_pssl_pretrained/model.pdparams), where the download links are also provided.

**(Optional) Pretraining**
---

There is no need to do this step if [STDC2](https://paddleseg.bj.bcebos.com/dygraph/pssl/stdc2_pssl_pretrained/model.pdparams) and [PPLite-Seg-B](https://paddleseg.bj.bcebos.com/dygraph/pssl/pp_liteseg_stdc2_pssl_pretrained/model.pdparams) are used.

Otherwise, checking the following steps for preparing the enviroment and the datasets:

1. Make sure that PaddleSeg is well installed. See the [installation guide](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/install.md) for details.

2. Make sure that [ImageNet](https://image-net.org/download-images.php) has been downloaded (~138G) and extracted to `data/ImageNet_org`. We only use the training set of ImageNet here. See below the dataset structure.

3. After getting the PSSL dataset link, download and extract to `data/pssl2.1_consensus`. See below the dataset structure.

Make sure that the datasets have structures as follows:

```
PaddleSeg
│   ...  
│
└───data
│   │
│   └───ImageNet_org
│   │   │  
│   │   └───train
│   │       │  
│   │       └───n01440764
│   │       │   │   n01440764_10026.JPEG
│   │       │   │   ...
│   │       │  
│   │       └───nxxxxxxxx
│   │  
│   └───pssl2.1_consensus  
│   │   │   imagenet_lsvrc_2015_synsets.txt
│   │   │   train.txt
│   │   └───train
│   │       │  
│   │       └───n01440764
│   │       │   │   n01440764_10026.JPEG_eiseg.npz
│   │       │   │   ...
│   │       │  
│   │       └───nxxxxxxxx
│   │       │   ...
```

Having installed PaddlePaddle and PaddleSeg and prepared datasets (ImageNet and PSSL), we can run the pre-training script:

* For STDC-Seg

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

path_save="work_dirs_stdc2_pssl"
python -m paddle.distributed.launch --log_dir $path_save tools/train.py \
       --config configs/pssl/stdc2_seg_pssl.yml \
       --log_iters 200 \
       --num_workers 12 \
       --save_interval 13345 \
       --keep_checkpoint_max 20 \
       --save_dir ${path_save}/snapshot
```

* For PP-LiteSeg

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

path_save="work_dirs_pp_liteseg_stdc2_pssl"
python -m paddle.distributed.launch --log_dir $path_save tools/train.py \
       --config configs/pssl/pp_liteseg_stdc2_pssl.yml \
       --log_iters 100 \
       --num_workers 12 \
       --save_interval 13345 \
       --keep_checkpoint_max 20 \
       --save_dir ${path_save}/snapshot
```

After the pre-training, the weights are saved in `${path_save}/snapshot/iter_xxx/model.pdparams`. Conventionall, we use the 5th epoch's checkpoint, i.e., 66725th iter (total batch size = 12 * 8 = 96), to do the downstream tasks.

For other models, modify the configuration file as needed.

## Downstream Tasks
---

For downstream tasks, PSSL does not need to change anything, except loading the pre-trained model. We can change this by simply adding one line in the config file, for example,

```yml
model:
  ...
  pretrained: work_dirs_pp_liteseg_stdc2_pssl/snapshot/iter_66725/model.pdparams
```


## Performance


### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|STDC2-Seg50|STDC2|512x512|40000|68.98%|70.07%|69.99%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/stdc2_seg_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/stdc2_seg_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=46d5d3cead36ee9d16df1d06b121b3bc) |
|STDC2-Seg50 + PSSL|STDC2|512x512|40000|74.49%|74.96%|75.79%|[model](https://paddleseg.bj.bcebos.com/dygraph/pssl/stdc2_voca_pssl/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pssl/stdc2_voca_pssl/train.log) |

### Cityscapes

| Model | Backbone | Training Iters | Train Crops | Test Resolution | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|-|
|PP-LiteSeg-B|STDC2|160000|1024x512|2048x1024|79.04%|79.52%|79.85%|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=12fa0144ca6a1541186afd2c53d31bcb)|
|PP-LiteSeg-B + PSSL|STDC2|160000|1024x512|2048x1024|79.06%|79.61%|79.97%|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k_pssl.yml)\|[model](https://paddleseg.bj.bcebos.com/dygraph/pssl/pplite_stdc2_cityscapes_pssl/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/pssl/pplite_stdc2_cityscapes_pssl/train.log) |
