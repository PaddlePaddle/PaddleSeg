# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

## Reference
> Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39, no. 12 (2017): 2481-2495.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SegNet|-|1024x512|80000|60.09%|60.88%|61.85%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segnet_cityscapes_1024×512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segnet_cityscapes_1024×512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=cb3abc86f6a3ebcd2d3033a68b23162d)|
