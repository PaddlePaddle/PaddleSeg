# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers

## Reference

> Zheng, Sixiao, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu et al. "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers." arXiv preprint arXiv:2012.15840 (2020).

## Performance

### CityScapes

| Model | Backbone | Head | Resolution | Training Iters | mIoU(slice) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Segmentation Transformer|Vision Transformer|Naive|769x769|40000|77.29%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_naive_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_naive_large_cityscapes_769x769_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=e21e3b4721366602a9a63c551108da1c)|
|Segmentation Transformer|Vision Transformer|PUP|769x769|40000|78.08%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_pup_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_pup_large_cityscapes_769x769_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=c25fdd4ac6221704d278b09f19ddf970) |
|Segmentation Transformer|Vision Transformer|MLA|769x769|40000|76.52%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_mla_large_cityscapes_769x769_40k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/setr_mla_large_cityscapes_769x769_40k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=993754909236b762b5276897ebec9c6d) |
