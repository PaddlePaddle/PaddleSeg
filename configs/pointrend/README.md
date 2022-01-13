# PointRend: Image Segmentation As Rendering

## Reference

> Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick. "PointRend: Image Segmentation As Rendering." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9799-9808. 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PointRend|ResNet50_vd|1024x512|80000|76.54%|76.84%|77.45%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pointrend_resnet50_os8_cityscapes_1024×512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/pointrend_resnet50_os8_cityscapes_1024×512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=bda232796400bc15141a088197d9a8c0) |

### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PointRend|ResNet50_vd|512x512|40000|72.82%|73.53%|74.62%|[model](hhttps://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/pointrend_resnet50_os8_voc12aug_512×512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/pointrend_resnet50_os8_voc12aug_512×512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=35c2d83707f51b23eabbe734606493a5) |
|PointRend|ResNet101_vd|512x512|40000|74.09%|74.7%|74.85%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/pointrend_resnet101_os8_voc12aug_512×512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/pointrend_resnet101_os8_voc12aug_512×512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=b2f7b7e99bba213db27b52826086686a) |
