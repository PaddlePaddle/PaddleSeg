# CCNet: Criss-cross attention for semantic segmentation

## Reference

> Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, Thomas S. Huang. "CCNet: Criss-cross attention for semantic segmentation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|CCNet|ResNet101_OS8|769x769|60000|80.95%|81.23%|81.32%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ccnet_resnet101_os8_cityscapes_769x769_60k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ccnet_resnet101_os8_cityscapes_769x769_60k/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6828616e27a1e15f1442beb3b4834048)|
