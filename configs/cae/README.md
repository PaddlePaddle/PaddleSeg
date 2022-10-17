# Context Autoencoder for Self-Supervised Representation Learning


## Reference

> Chen, Xiaokang, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. "Context autoencoder for self-supervised representation learning." arXiv preprint arXiv:2202.03026 (2022).

## Performance

### ADE20k
| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|UPerNetCAE|CAE|512x512|160000| 49.69% | - | - |[model](https://bj.bcebos.com/paddleseg/dygraph/ade20k/upernet_caebase_ade20k_512x512_160k/model.pdparams)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/upernet_caebase_ade20k_512x512_160k/train.log)\| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=46b862c422fb5a9b5b12d00472527ffd) |
