# Panoptic Feature Pyramid Networks

## Reference

> Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár. "Panoptic Feature Pyramid Networks." arXiv preprint arXiv:1901.02446(2019).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PFPNNet|ResNet101_vd|1024x512|40000|79.07%|79.46%|79.75%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pfpn_resnet101_os8_cityscapes_512x1024_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pfpn_resnet101_os8_cityscapes_512x1024_40k/train.log )\| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=)|
