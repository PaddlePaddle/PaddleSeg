# Lightweight and Progressively-Scalable Networks for Semantic Segmentation

## Reference

> Zhang, Yiheng and Yao, Ting and Qiu, Zhaofan and Mei, Tao. "Lightweight and Progressively-Scalable Networks for Semantic Segmentation."
arXiv preprint arXiv:2207.13600 (2022).

## Performance

### ImageNet pretrained

Trained medium model of LPSNet on ImageNet dataset for 100 epochs, finally got top-1 accuracy **0.543** and top-5 accuracy **0.786**.

Training settings are as following.

**Optimizer**

| Optimizer    | LR Schedular                      | Regularizer  |
| ------------ | --------------------------------- | ------------ |
| momentum:0.9 | Piecewice                         | L2           |
| -            | lr:0.1                            | coeff:0.0001 |
| -            | decay_epochs:[30, 60, 90]         | -            |
| -            | values:[0.1, 0.01, 0.001, 0.0001] | -            |

**Data augments**

| RandCropImage | RandFlipImage   | Normalize             |
| ------------- | --------------- | --------------------- |
| size: 224     | flip horizontal | ImageNet mean and std |

With the pretrained model, LPSNet is able to obtain **74.28%** mIoU on CityScapes eval set, whose traning settings followed description in the official paper.

Furthermore, in order to obtain the best performance of semantic segmentation and accelerate convergence speed, more complicated data augments are applied, for instance, **random erasing** and batch transform **mixup**.

The performance of classification improved a little, whose top-1 accuracy increased to **0.564** and top-5 accuracy increased to **0.805**. However, the model are unable to converge to higher than before.

So the following performance are all trained with the first ImageNet pretrained weights.

###     Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|lpsnet_m|-|1536x769|200000|75.29%|76.03%|77.03%|[model]() \| [log]() \| [vdl]()|

To make up for the deficiencies of such a low-performance pretrained model, the following attempts are tested:

1. Extend interations to **200k**.
2. Adjust the parameters of **RandomDistort**.
3. Decrease the threshold of **OhemCrossEntropyLoss.**
4. Appropriately increase the **learning rate**.

There are plenty of ways to perfect model performance. But with the limited time and computational resources, only a few have been tested. The result is, way 1 and 2 are proven to be valid.
