# Asymmetric Non-local Neural Networks for Semantic Segmentation

## Reference

> Zhu, Zhen, Mengde Xu, Song Bai, Tengteng Huang, and Xiang Bai. "Asymmetric non-local neural networks for semantic segmentation." In Proceedings of the IEEE International Conference on Computer Vision, pp. 593-602. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ANN|ResNet50_OS8|1024x512|80000|79.09%|79.31%|79.56%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ann_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ann_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=b849c8e06b6ccd33514d436635b9e102)|
|ANN|ResNet101_OS8|1024x512|80000|80.61%|80.98%|81.25%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ann_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ann_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ed1cb9321385f1480dda418db71bd4c0)|
