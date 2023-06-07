# SeaFormer: Squeeze-Enhanced Axial Transformer For Mobile Semantic Segmentation

## Reference

> Qiang Wan, Zilong Huang, Jiachen Lu, Gang Yu and Li Zhang. "SeaFormer: Squeeze-Enhanced Axial Transformer For Mobile Semantic Segmentation." Published as a conference paper at ICLR 2023, arXiv:2301.13156v4.

## Performance

### ADE20k

| Model | size | Head | Resolution | Training Iters | mIoU (slice) | Links |
| :-:   | :-:  | :-:        | :-:            | :-:          | :-:          | :-:          |
| SeaFormer | tiny | LightHead | 512x512 | 160000 | 34.58 | [model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_tiny_ade20k_512x512_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/seaformer_tiny_ade20k_512x512_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=72cba29a94f195914b47554466a1e7d8) |
| SeaFormer | base | LightHead | 512x512 | 160000 | 40.92 | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/seaformer_ade20k_512x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/seaformer_ade20k_512x512_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=f43d0185995605fb99da6597121a96f1) |
