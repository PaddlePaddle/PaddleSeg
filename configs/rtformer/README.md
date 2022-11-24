# RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer

## Reference

> Wang, Jian, Chenhui Gou, Qiman Wu, Haocheng Feng, Junyu Han, Errui Ding, and Jingdong Wang. "RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer." arXiv preprint arXiv:2210.07124 (2022).

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|RTFormer-Base|-|1024x512|120000|79.24%|79.80%|80.19%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/rtformer_base_cityscapes_1024x512_120k/model.pdparams) \| [log]() \| [vdl]()|
|RTFormer-Slim|-|1024x512|120000|76.31%|77.05%|77.58%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/rtformer_slim_cityscapes_1024x512_120k/model.pdparams) \| [log]() \| [vdl]()|


### ADE20k

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|RTFormer-Base|-|512x512|160000|42.02%|42.43%|42.72%|[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/rtformer_base_ade20k_512x512_160k/model.pdparams) \| [log]() \| [vdl]()|
|RTFormer-Slim|-|512x512|160000|36.67%|37.32%|37.20%|[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/rtformer_slim_ade20k_512x512_160k/model.pdparams) \| [log]() \| [vdl]()|
