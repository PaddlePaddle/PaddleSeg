# LRASPP

## Reference

> Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, et al. "Searching for MobileNetV3." In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314-1324. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Pooling Method | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|LRASPP|MobileNetV3_large_x1_0_os8|1024x512|Global|80000|72.33%|72.63%|73.77%|[config](./lraspp_mobilenetv3_cityscapes_1024x512_80k.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app?id=d42c84fe5407fd2f1cf08e355348c441)|
|LRASPP|MobileNetV3_large_x1_0_os8|1024x512|Large kernel|80000|73.19%|73.40%|74.20%|[config](lraspp_mobilenetv3_cityscapes_1024x512_80k_large_kernel.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k_large_kernel/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k_large_kernel/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=76c9c025d913c90ba703eeb5cef307e1)|
|LRASPP|MobileNetV3_large_x1_0|1024x512|Global|80000|70.13%|70.43%|72.12%|[config](lraspp_mobilenetv3_cityscapes_1024x512_80k_os32.yml) \| [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k_os32/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/lraspp_mobilenetv3_cityscapes_1024x512_80k_os32/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=2ee4619b2858f38ff92cf602b793d248)|

Note that:
- The *global* pooling method refers to the use of a global average pooling layer in the LR-ASPP head, which easily adapts to small-sized input images. In contrast, the *large-kernel* pooling method uses a 49x49 kernel for average pooling, which is consistent with the design in the original paper.
- MobileNetV3_\*_os8 is a variant of MobileNetV3 tailored for semantic segmentation tasks. The output stride is 8, and dilated convolutional layers are used in place of the vanilla convolutional layers in the last two stages.
