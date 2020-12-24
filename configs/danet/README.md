# Dual Attention Network for Scene Segmentation

## Reference

> Fu J, Liu J, Tian H, et al. Dual attention network for scene segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 3146-3154.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DANet|ResNet50_OS8|1024x512|80000|80.27%|-|-|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/danet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/danet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6caecf1222a0cc9124a376284a402cbe)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DANet|ResNet50_OS8|1024x512|40000|78.55%|-|-|[model](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/danet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/danet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=51a403a54302bc81dd5ec0310a6d50ba)|
