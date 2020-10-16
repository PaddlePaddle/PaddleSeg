# Deep High-Resolution Representation Learning for Visual Recognition

## Reference
> Wang J, Sun K, Cheng T, et al. Deep high-resolution representation learning for visual recognition[J]. IEEE transactions on pattern analysis and machine intelligence, 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (multi-scale) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|HRNet_W18|1024x512|80000|80.34%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/fcn_hrnetw18_cityscapes_1024x512_80k.yml/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/fcn_hrnetw18_cityscapes_1024x512_80k.yml/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=141ed1c7aa77474ec2a2d063713570f9)|
|FCN|ResNet101_OS8|1024x512|160000|81.17%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/fcn_hrnetw48_cityscapes_1024x512_80k.yml/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/fcn_hrnetw48_cityscapes_1024x512_80k.yml/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=6f219d4b9bab266385ab6023ea097aa6)|
