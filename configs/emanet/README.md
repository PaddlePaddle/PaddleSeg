# Expectation-Maximization Attention Networks for Semantic Segmentation

## Reference

> Xia Li, Zhisheng Zhong, Jianlong Wu, Yibo Yang, Zhouchen Lin, Hong Liu:
Expectation-Maximization Attention Networks for Semantic Segmentation. ICCV 2019: 9166-9175.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) |Links |
|-|-|-|-|-|-|-|-|
|EMANet|ResNet50_OS8|1024x512|80000|77.58%|77.98%|78.23%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3e053a214d60822d6e65445b8614d052)|
|EMANet|ResNet101_OS8|769x769|80000|79.42%|79.83%|80.33%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=87be6389cdada711f5c6ada21d9ef6cd)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|EMANet|ResNet50_OS8|512x512|40000|78.79%|78.90%|79.17%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3e60b80b984a71f3d2b83b8a746a819c)|
|EMANet|ResNet101_OS8|512x512|40000|79.73%|79.97%| 80.67%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=f33479772409766dbc40b5f031cbdb1a)|
