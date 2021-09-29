# PointRend: Image Segmentation As Rendering

## Reference

> Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick. "PointRend: Image Segmentation As Rendering." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9799-9808. 2020.

## Performance

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PointRend|ResNet50_vd|1024x512|80000|76.54%|76.84%|77.45%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pointrend_resnet50_os8_cityscapes_1024×512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pointrend_resnet50_os8_cityscapes_1024×512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=bda232796400bc15141a088197d9a8c0) |
