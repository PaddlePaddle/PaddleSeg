# Graph-Based Global Reasoning Networks

## Reference

> Chen, Yunpeng, Marcus Rohrbach, Zhicheng Yan, Yan Shuicheng, Jiashi Feng, and Yannis Kalantidis. "Graph-based global reasoning networks." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 433-442. 2019.


## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GloRe|ResNet50_OS8|1024x512|80000|78.26%|78.61%|78.72%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/glore_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/glore_resnet50_os8_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=de754e39ac9de4d2e951915c2334d6ec) |


### Pascal VOC 2012 + Aug

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GloRe|ResNet50_OS8|512x512|40000|80.16%|80.35%|80.40%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/glore_resnet50_os8_voc12aug_512x512_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/glore_resnet50_os8_voc12aug_512x512_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e40c1dd8d4fcbf2dcda01242dec9d9b5) |
