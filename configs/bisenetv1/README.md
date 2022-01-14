# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

## Reference

> Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation." In Proceedings of the European conference on computer vision (ECCV), pp. 325-341. 2018.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|BiSeNetV1|-|1024x512|160000|75.19%|75.99%|76.77%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenetv1_cityscapes_1024x512_160k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenetv1_cityscapes_1024x512_160k/train.log)\|[vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=d2807bd39677b369ee84054e46a3df96)|
