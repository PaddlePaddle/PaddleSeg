# Asymmetric Non-local Neural Networks for Semantic Segmentation

## Reference

> Zhu, Zhen, Mengde Xu, Song Bai, Tengteng Huang, and Xiang Bai. "Asymmetric non-local neural networks for semantic segmentation." In Proceedings of the IEEE International Conference on Computer Vision, pp. 593-602. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (multi-scale) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ANN|ResNet50_OS8|1024x512|160000|78.13%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/ann_resnet50_os8_cityscapes_1024x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ann_resnet50_os8_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e5e8fb0c5d8c81558981bcf0b403af3f)|
|ANN|ResNet101_OS8|1024x512|160000|80.25%|-|[model](https://paddleseg.bj.bcebos.com/dygraph/ann_resnet101_os8_cityscapes_1024x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ann_resnet101_os8_cityscapes_1024x512_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=7c8bf49eeb74a02f978b4050ebbea03c)|
