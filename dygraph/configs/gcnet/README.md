# GCNet: Non-local networks meet squeeze-excitation networks and beyond

## Reference

> Cao, Yue, Jiarui Xu, Stephen Lin, Fangyun Wei, and Han Hu. "GCNet: Non-local networks meet squeeze-excitation networks and beyond." In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 0-0. 2019.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GCNet|ResNet50_OS8|1024x512|80000|79.50%|79.77%|79.69%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/gcnet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/gcnet_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e3801edb9a6f5b33eb890f5a1ae6ed7b)|
|GCNet|ResNet101_OS8|1024x512|80000|81.01%|81.30%|81.64%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/gcnet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/gcnet_resnet101_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=aa88e7980f4d6839537662a3a3d18851)|
