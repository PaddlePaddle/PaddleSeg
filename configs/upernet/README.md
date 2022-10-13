# Unified Perceptual Parsing for SceneUnderstanding


## Reference

> Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun. "Unified Perceptual Parsing for Scene Understanding." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|UPerNet|ResNet101_OS8|512x1024|40000|79.58%|80.11%|80.41%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/upernet_resnet101_os8_cityscapes_512x1024_40k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/upernet_resnet101_os8_cityscapes_512x1024_40k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=c635ae2e70e148796cd58fae5273c3d6)|


### ADE20k
| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|UPerNetCAE|CAE|512x512|160000|49.69% | - | - |[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/upernet_caebase_ade20k_512x512_160k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/upernet_resnet101_os8_cityscapes_512x1024_40k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=c635ae2e70e148796cd58fae5273c3d6)|
