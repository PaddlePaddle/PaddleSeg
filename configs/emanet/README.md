# Expectation-Maximization Attention Networks for Semantic Segmentation

## Reference

> Xia Li, Zhisheng Zhong, Jianlong Wu, Yibo Yang, Zhouchen Lin, Hong Liu:
Expectation-Maximization Attention Networks for Semantic Segmentation. ICCV 2019: 9166-9175.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) |Links |
|-|-|-|-|-|-|-|-|
|EMANet|ResNet50_OS8|1024x512|80000|79.05%|79.34%|79.69%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=0a05a0c4cd7d785b9707bdc59f55f585)|
|EMANet|ResNet101_OS8|1024x512|80000|80.00%|80.23%|80.53%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/emanet_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ee6926322b8e292ce23ce62ecdaa3439)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|EMANet|ResNet50_OS8|512x512|40000|78.60%|78.90%|79.17%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3e60b80b984a71f3d2b83b8a746a819c)|
|EMANet|ResNet101_OS8|512x512|40000|79.47%|79.97%| 80.67%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=f33479772409766dbc40b5f031cbdb1a)|
