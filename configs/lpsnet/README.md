# Lightweight and Progressively-Scalable Networks for Semantic Segmentation

## Reference

> Zhang, Yiheng and Yao, Ting and Qiu, Zhaofan and Mei, Tao. "Lightweight and Progressively-Scalable Networks for Semantic Segmentation."
arXiv preprint arXiv:2207.13600 (2022).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|lpsnet_s|-|1536x769|200000|71.73%|72.71%|73.61%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_s_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_s_cityscapes_1536x1024_200k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a1f70216053d015234de95bcbe2201ff)|
|lpsnet_m|-|1536x769|200000|75.29%|76.03%|77.12%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_m_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_m_cityscapes_1536x1024_200k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=b4fad2f53cce38392ebc5821ff577b4c)|
|lpsnet_l|-|1536x769|200000|75.72%|76.53%|77.50%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_l_cityscapes_1536x1024_200k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lpsnet_l_cityscapes_1536x1024_200k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e70029f7b1f40007091cf5da58777b69)|

Note that: Since the original paper does not provide all the implementation details, nor release the official training code, we first pre-trained the models on the ImageNet dataset, and then fine-tuned the models on the Cityscapes dataset. Specifically, compared with the experimental settings in the original paper, we made two significant changes during fine-tuning to obtain the above results:

1. Extend iterations to **200k**.
2. Adjust the parameters used in **color jittering**.

For the pre-training results of the models on the ImageNet dataset, please see the [ImageNet](#ImageNet) section, where we also have a discussion on the impact of the pre-trained model.

### ImageNet

| Model    | Epoch | Top-1 accuracy | Top-5 accuracy |
| -------- | ----- | -------------- | -------------- |
| lpsnet_s | 120   | 0.403          | 0.666          |
| lpsnet_m | 100   | 0.543          | 0.786          |
| lpsnet_l | 120   | 0.525          | 0.773          |

Training settings are as following.

**Optimizer**

| Optimizer          | LR Scheduler                       | Regularizer  |
| ------------------ | ---------------------------------- | ------------ |
| type: Momentum     |  type: Piecewice                   | type: L2     |
| momentum: 0.9      | lr: 0.1                            | coeff:0.0001 |
| use_nesterov: true | decay_epochs: [30, 60, 90]         | -            |
| -                  | values: [0.1, 0.01, 0.001, 0.0001] | -            |

**Data Augmentation**

| RandCropImage | RandFlipImage   | Normalize             |
| ------------- | --------------- | --------------------- |
| size: 224     | flip horizontal | ImageNet mean and std |

With the pre-trained model, lpsnet_m is able to obtain **74.28%** mIoU on Cityscapes val set under the same experimental settings as the original paper.

Further more, we also tried more complicated data augmentation strategies, e.g. **random erasing** and **batched mix-up**. The performance of classification improved by a considerable margin: on the ImageNet dataset, the top-1 accuracy increased to **0.564** and the top-5 accuracy increased to **0.805**. However, we did not observe better segmentation performance of the model on the Cityscapes dataset using these pre-trained models. Therefore, these data augmentation strategies were not used.
