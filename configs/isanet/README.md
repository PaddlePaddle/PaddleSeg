# Interlaced Sparse Self-Attention for Semantic Segmentation

## Reference

> CLang Huang, Yuhui Yuan, Jianyuan Guo, Chao Zhang, Xilin Chen, Jingdong Wang: Interlaced Sparse Self-Attention for Semantic Segmentation.  Conference on Computer Vision and Pattern Recognition (CVPR2020).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | Links |
|-|-|-|-|-|-|
|ISANet|ResNet50_OS8|769x769|80000|79.03%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/isanet_resnet50_os8_cityscapes_769x769_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/isanet_resnet50_os8_cityscapes_769x769_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ab7cc0627fdbf1e210557c33d94d2e8c)|
|ISANet|ResNet101_OS8|769x769|80000|80.10%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/isanet_resnet101_os8_cityscapes_769x769_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/isanet_resnet101_os8_cityscapes_769x769_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=76366b80293c3ac2374d981b4573eb52)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | Links |
|-|-|-|-|-|-|
|ISANet|ResNet50_OS8|512x512|40000|79.69%|[model](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/isanet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/isanet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=84af8df983e48f1a0c89154a26f55032)|
|ISANet|ResNet101_OS8|512x512|40000|79.57%|[model](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/emanet_resnet101_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6874531f0adbfc72f22fb816bb231a46)|