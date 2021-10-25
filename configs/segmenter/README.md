# Segmenter: Transformer for Semantic Segmentation

## Reference

> Strudel, Robin, et al. "Segmenter: Transformer for Semantic Segmentation." arXiv preprint arXiv:2105.05633 (2021).

## Performance

### ADE20k

| Model | Backbone | Decoder | Patch Size | Resolution | Training Iters | mIoU(slice) | Links |
| :-:   | :-:      | :-:     | :-:        | :-:        | :-:            | :-:         | :-:   |
| Segmenter | ViT small | Linear  | 16 | 512*512 | 80000 | 45.48 | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_small_linear_ade20k_512x512_160k/model.pdparams) | [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_small_linear_ade20k_512x512_160k/train.log) | [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=4dc954a9b774e4807c07c511c04ce0f6) |
| Segmenter | ViT small | Mask    | 16 | 512*512 | 80000 | 45.15 | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_small_mask_ade20k_512x512_160k/model.pdparams) | [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_small_mask_ade20k_512x512_160k/train.log) | [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0fdd5191ecec56bbdf08259cc6c32a21) |
| Segmenter | ViT base  | Linear  | 16 | 512*512 | 80000 | 48.13 | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_base_linear_ade20k_512x512_160k/model.pdparams) | [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_base_linear_ade20k_512x512_160k/train.log) | [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=992f38b3f937de87dc74a888d217f53e) |
| Segmenter | ViT base  | Mask    | 16 | 512*512 | 80000 | 48.49 | [model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_base_mask_ade20k_512x512_160k/model.pdparams) | [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/segmenter_vit_base_mask_ade20k_512x512_160k/train.log) | [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=16a7380069b6435bdf6e566dcc7f4a6b) |
