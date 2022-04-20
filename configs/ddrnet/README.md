# Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes

## Reference

> Yuanduo Hong, Huihui Pan, Weichao Sun, Yisong Jia. "Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes." arXiv preprint arXiv:2101.06085 (2021).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|DDRNet_23|-|1024x1024|120000|79.85%|80.11%|80.44%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ddrnet23_cityscapes_1024x1024_120k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ddrnet23_cityscapes_1024x1024_120k/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=33c0d5f37e5a708c605e43ef3845ea56)|
