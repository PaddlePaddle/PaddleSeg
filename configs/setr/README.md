# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers

## Reference

> Zheng, Sixiao, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu et al. "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers." arXiv preprint arXiv:2012.15840 (2020).

## Performance

### CityScapes

| Model | Backbone | Head | Resolution | Training Iters | mIoU(slice) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Segmentation Transformer|Vision Transformer|Naive|769x769|40000|77.29%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_naive_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_naive_large_cityscapes_769x769_40k/train.log) \| [vdl]()|
|Segmentation Transformer|Vision Transformer|PUP|769x769|40000|78.08%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_pup_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_pup_large_cityscapes_769x769_40k/train.log) \| [vdl]() |
|Segmentation Transformer|Vision Transformer|MLA|769x769|40000|76.52%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_mla_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_mla_large_cityscapes_769x769_40k/train.log) \| [vdl]() |
