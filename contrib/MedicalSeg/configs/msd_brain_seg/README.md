# [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
The Medical Segmentation Decathlon is a collection of medical image segmentation datasets. It contains a total of 2,633 three-dimensional images collected across multiple anatomies of interest, multiple modalities and multiple sources. Specifically, it contains data for the following body organs or parts: Brain, Heart, Liver, Hippocampus, Prostate, Lung, Pancreas, Hepatic Vessel, Spleen and Colon.
## Performance


### Unetr
>   Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger Roth, Daguang Xu ·  "UNETR: Transformers for 3D Medical Image Segmentation" Accepted to IEEE Winter Conference on Applications of Computer Vision (WACV) 2022

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|128x128x128|1e-4|30000|71.8%|[model](https://bj.bcebos.com/paddleseg/paddleseg/medicalseg/msd_brain_seg/unetr_msd_brain_seg_1e-4/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/paddleseg/medicalseg/msd_brain_seg/unetr_msd_brain_seg_1e-4/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=04e012eef21ea8478bdc03f9c5b1032f)|
