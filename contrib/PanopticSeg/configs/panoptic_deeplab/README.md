# Panoptic-Deeplab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

## Reference

> Cheng, Bowen, et al. "Panoptic-deeplab: A simple, strong, and fast baseline for bottom-up panoptic segmentation." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | PQ | mIoU | mAP50 | Links |
|-|-|-|-|-|-|-|-|
|Panoptic-DeepLab|ResNet50-vd|1025x513|90k|60.32%|46.34%|79.68%|[config](panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/cityscapes/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k/model.pdparams)|

+ *The models were trained using 8 GPUs.*
+ *We observed better performance by using the input resolution of 1025x513 instead of 1024x512 and setting `align_corners` to True during training.*