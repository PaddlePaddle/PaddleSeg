# Vision Transformer Adapter for Dense Predictions

## Reference

> Chen, Zhe, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. "Vision Transformer Adapter for Dense Predictions." arXiv preprint arXiv:2205.08534 (2022).

## Performance

### ADE20K

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|UPerNetViTAdapter|ViT-Adapter-Tiny|512x512|160000|41.90%|-|-|[model](https://paddleseg.bj.bcebos.com/dygraph/ade20k/upernet_vit_adapter_tiny_ade20k_512x512_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/ade20k/upernet_vit_adapter_tiny_ade20k_512x512_160k/train_log.txt) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=88173046bd09f61da5f48db66baddd7d)|
