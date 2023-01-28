# Per-Pixel Classification is Not All You Need for Semantic Segmentation

## Reference

> Cheng, Bowen, Alex Schwing, and Alexander Kirillov. "Per-pixel classification is not all you need for semantic segmentation." Advances in Neural Information Processing Systems 34 (2021): 17864-17875.

## Performance

### ADE20k

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Maskformer-tiny|SwinTransformer|512x512|160000|47.93|-|-|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_tiny/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_tiny/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=fd734e48cac51de1f6a04624567caed9)|
|Maskformer-small|SwinTransformer|512x512|160000|50.4|-|-|[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_small/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/maskformer_ade20k_swin_small/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=a5809bed3685e61680b84c4b5a88148c)|

* Maskformer support different network setting including tiny, small, base and large. The training result of base and large is not provided, but it should be consistent with the paper

* Maskformer-Base and Maskformer-Large will be evaled with multi-scale and flip as the original codebase .
