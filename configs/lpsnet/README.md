# Lightweight and Progressively-Scalable Networks for Semantic Segmentation

## Reference

> Zhang, Yiheng and Yao, Ting and Qiu, Zhaofan and Mei, Tao. "Lightweight and Progressively-Scalable Networks for Semantic Segmentation."
arXiv preprint arXiv:2207.13600 (2022).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|lpsnet_m|-|1536x769|200000|75.29%|76.03%|77.03%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_m_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_m_cityscapes_1536x1024_200k/train.log) \| [vdl]()|
|lpsnet_s|-|1536x769|200000|71.73%|72.71%|73.76%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_s_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_s_cityscapes_1536x1024_200k/train.log) \| [vdl]()|
|lpsnet_l|-|1536x769|200000|75.72%|76.53%|77.23%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_l_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_l_cityscapes_1536x1024_200k/train.log) \| [vdl]()|

Note that: To make up for the deficiencies of low-performance pretrained(see [ImageNet](#ImageNet)) models, the following changes are applied:

1. Extend iterations to **200k**.
2. Adjust the parameters of **RandomDistort**.

### ImageNet

| Model    | Epoch | Top-1 accuracy | Top-5 accuracy |
| -------- | ----- | -------------- | -------------- |
| lpsnet_s | 120   | 0.403          | 0.666          |
| lpsnet_m | 100   | 0.543          | 0.786          |
| lpsnet_l | 120   | 0.525          | 0.773          |

Training settings are as following.

**Optimizer**

| Optimizer          | LR Schedular                       | Regularizer  |
| ------------------ | ---------------------------------- | ------------ |
| type: Momentum     |  type: Piecewice                   | type: L2     |
| momentum: 0.9      | lr: 0.1                            | coeff:0.0001 |
| use_nesterov: true | decay_epochs: [30, 60, 90]         | -            |
| -                  | values: [0.1, 0.01, 0.001, 0.0001] | -            |

**Data Augmentation**

| RandCropImage | RandFlipImage   | Normalize             |
| ------------- | --------------- | --------------------- |
| size: 224     | flip horizontal | ImageNet mean and std |

With the pretrained model, lpsnet_m is able to obtain **74.28%** mIoU on CityScapes eval set, whose training settings followed description in the official paper.

Furthermore, in order to obtain the best performance of semantic segmentation and accelerate convergence speed, more complicated data augments are applied, for instance, **random erasing** and batch transform **mixup**.

The performance of classification improved a little, whose top-1 accuracy increased to **0.564** and top-5 accuracy increased to **0.805**. However, the model is unable to converge to higher than before.

So all the models in section [Cityscapes](#Cityscapes) are all pretrained with the original ImageNet training settings.
