# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers

## Reference

> Strudel, Robin, et al. "Segmenter: Transformer for Semantic Segmentation." arXiv preprint arXiv:2105.05633 (2021).

## Performance

### CityScapes

| Model | Backbone | Head | Resolution | Training Iters | mIoU(slice) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Segmentation Transformer|Vision Transformer|Naive|768x768|40000|77.29%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_naive_large_cityscapes_769x769_40k/model.pdparams) \| [log]() \| [vdl]()|


| Model | Backbone | Patch Size | Head | Resolution | Training Iters | mIoU(slice) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Segmenter | ViT base  | 16 | mask_transformer | 768*768 | 80000 |
| Segmenter | ViT large | 16 | mask_transformer | 768*768 | 80000 |
