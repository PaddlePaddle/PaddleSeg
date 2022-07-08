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

Having prepared datasets (ImageNet and PSSL) and installed PaddlePaddle and PaddleSeg, we can run the pre-training script:

* For STDC-Seg

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

path_save="work_dirs_stdc2_pssl"
fleetrun --log_dir $path_save train.py \
       --config configs/stdcseg/stdc2_seg_pssl.yml \
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
fleetrun --log_dir $path_save train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_pssl.yml \
       --log_iters 100 \
       --num_workers 12 \
       --save_interval 13345 \
       --keep_checkpoint_max 20 \
       --save_dir ${path_save}/snapshot
```

After the pre-training, the weights are saved in `${path_save}/snapshot/iter_xxx/model.pdparams`. Conventionall, we use the 5th epoch's checkpoint to do the downstream tasks, i.e., 66725th iter.

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
