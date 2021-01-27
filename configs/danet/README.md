# Dual Attention Network for Scene Segmentation

## Reference

> Fu, Jun, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, and Hanqing Lu. "Dual attention network for scene segmentation." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3146-3154. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DANet|ResNet50_OS8|1024x512|80000|80.27%|80.53%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/danet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/danet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6caecf1222a0cc9124a376284a402cbe)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DANet|ResNet50_OS8|512x512|40000|78.55%|78.93%|79.68%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/danet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/danet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=51a403a54302bc81dd5ec0310a6d50ba)|
