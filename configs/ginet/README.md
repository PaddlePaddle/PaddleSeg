# GINet: Graph Interaction Network for Scene Parsing

## Reference

> Wu, Tianyi, Yu Lu, Yu Zhu, Chuang Zhang, Ming Wu, Zhanyu Ma, and Guodong Guo. "GINet: Graph interaction network for scene parsing." In European Conference on Computer Vision, pp. 34-51. Springer, Cham, 2020.


## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GINet|ResNet50_OS8|1024x512|80000|78.66%|79.07%|79.2%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ginet_resnet50_os8_cityscapes_1024×512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ginet_resnet50_os8_cityscapes_1024×512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=bb439dc87b311c4105c426eadd5a641e) |
|GINet|ResNet101_OS8|1024x512|80000|78.4%|78.72%|78.99%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ginet_resnet101_os8_cityscapes_1024×512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ginet_resnet101_os8_cityscapes_1024×512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ffae8d094b755a4313d6e02540de9515) |

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GINet|ResNet50_OS8|512x512|40000|81.97%|82.02%|81.65%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ginet_resnet50_os8_voc12aug_512×512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ginet_resnet50_os8_voc12aug_512×512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=638ff8bcc88575489ee36da0edad51b6) |
|GINet|ResNet101_OS8|512x512|40000|79.79%|79.99%|80.6%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ginet_resnet101_os8_voc12aug_512×512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ginet_resnet101_os8_voc12aug_512×512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a1f7d1040f371585d9aac1610116f594) |

### ADE20K

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GINet|ResNet50_OS8|520x520|150000|43.45%|43.98%|43.80%|[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/ginet_resnet50_os8_ade20k_520x520_150k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/ginet_resnet50_os8_ade20k_520x520_150k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=665901e12a35319710197380a5dfafa5) |
|GINet|ResNet101_OS8|520x520|150000|45.79%|45.94%|46.18%|[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/ginet_resnet101_os8_ade20k_520x520_150k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/ginet_resnet101_os8_ade20k_520x520_150k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=46b63c18e421e2a0ba95faefdc8d5c39) |
