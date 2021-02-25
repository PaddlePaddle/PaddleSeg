# Deep High-Resolution Representation Learning for Visual Recognition

## Reference
> Wang, Jingdong, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu et al. "Deep high-resolution representation learning for visual recognition." IEEE transactions on pattern analysis and machine intelligence (2020).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|HRNet_W18|1024x512|80000|78.97%|79.49%|79.74%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=bebec8e1a3802c4babd3c69e1bf50d51)|
|FCN|HRNet_W48|1024x512|80000|80.70%|81.24%|81.56%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ae1cb76014cdc54406c36f1e3dc2a530)|

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN|HRNet_W18|512x512|40000|75.39%|76.04%|77.09%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw18_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw18_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=fbe6caaca0f7d7ea1dba1c60b8db2a7e)|
|FCN|HRNet_W48|512x512|40000|78.72%|79.52%|80.10%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw48_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw48_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=20b404212fcbb5b7b329ab0c16124553)|
