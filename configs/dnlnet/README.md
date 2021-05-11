# Disentangled Non-Local Neural Networks

## Reference

> Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, Han Hu:
Disentangled Non-local Neural Networks. ECCV (15) 2020: 191-207.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) |Links |
|-|-|-|-|-|-|-|-|
|DNLNet|ResNet50_OS8|1024x512|80000|79.95%|80.43%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/dnlnet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/dnlnet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=922cf0682c5e684507ab54a14ef12847)|
|DNLNet|ResNet101_OS8|1024x512|80000|81.03%|81.38%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/dnlnet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/dnlnet_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3e0d13c4d9dbf4115bbba2abdc88122c)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DNLNet|ResNet50_OS8|512x512|40000|80.89%|81.31%|81.56%|[model](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/dnlnet_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/dnlnet_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=8877c77bef8b227af22c5eb3017138ce)|
|DNLNet|ResNet101_OS8|512x512|40000|80.49%|80.83%| 81.33%|[model](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/dnlnet_resnet101_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/dnlnet_resnet101_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1d42c22da1c465d9a38e4204bebeeb54)|
