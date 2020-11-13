# Deep High-Resolution Representation Learning for Visual Recognition

## Reference
> Wang J, Sun K, Cheng T, et al. Deep high-resolution representation learning for visual recognition[J]. IEEE transactions on pattern analysis and machine intelligence, 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|HRNet_W18|1024x512|80000|78.97%|79.49%|79.74%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=bebec8e1a3802c4babd3c69e1bf50d51)|
|FCN|HRNet_W48|1024x512|80000|80.70%|81.24%|81.56%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ae1cb76014cdc54406c36f1e3dc2a530)|
