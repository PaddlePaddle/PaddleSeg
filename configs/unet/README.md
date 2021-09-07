# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Reference
> Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | Batch Size | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|UNet|-|1024x512|160000|4|65.00%|66.02%|66.89%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=67b3338de34ad09f0cb5e7c6856305cc)|
