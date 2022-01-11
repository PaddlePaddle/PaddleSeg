# ENCNet: Context Encoding for Semantic Segmentation

## Reference
> Hang Zhang, Kristin Dana, et, al. "Context Encoding for Semantic Segmentation". In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 7151-7160. 2018.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ENCNet|ResNet101_vd|1024x512|80000|79.42%|80.02%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/encnet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/encnet_resnet101_os8_cityscapes_1024x512_80k/train.log )\| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=c2b819e6b666e4e50bba4b525f515d41)|
