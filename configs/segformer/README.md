# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

## Reference

> Xie, Enze, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU(slice) | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|SegFormer_B0|-|1024x1024|160000|76.73%|77.16%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b0_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b0_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=227e067add44d44383c402ec5aead11b)|
|SegFormer_B1|-|1024x1024|160000|78.35%|78.64%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b1_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b1_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a0f4e8eacf346826e3150989b6a9f849)|
|SegFormer_B2|-|1024x1024|160000|81.60%|81.82%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b2_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b2_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=734c0d99d858d0db7ff58f03d18289fe)|
|SegFormer_B3|-|1024x1024|160000|82.47%|82.60%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b3_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b3_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=406282a64c45d008bf4445c5669d6579)|
|SegFormer_B4|-|1024x1024|160000|82.38%|82.59%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b4_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b4_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=dc51a262eb8be9273970354ed445e760)|
|SegFormer_B5|-|1024x1024|160000|82.58%|82.82%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b5_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b5_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=306d042a8e4d82ccceabd988a478a2f8)|
