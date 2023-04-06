# PSA: Polarized Self-Attention: Towards High-quality Pixel-wise Regression

## Reference

> Huajun Liu, Fuqiang Liu, Xinyi Fan and Dong Huang. "Polarized Self-Attention: Towards High-quality Pixel-wise Regression." arXiv preprint arXiv:2107.00782v2 (2021).

## Performance

### Cityscapes

|      Model       |    Backbone     | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |                            Links                             |
| :--------------: | :-------------: | :--------: | :------------: | :----: | :---------: | :------------: | :----------------------------------------------------------: |
| OCRNet-HRNet+psa | HRNETV2_W48+psa | 1024x2048  |     150000     | 84.62% |   84.90%    |     84.01%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k/model.pdparams)\|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k/train.log)\|[vdl](#) |

### Notes

* This is the MscaleOCRNet that supports PSA.
* Since we cannot reproduce the training results from [the authors&#39; official repo](https://github.com/DeLightCMU/PSA), we follow the settings in the original paper to train and evaluate our models, and the final accuracy is lower than that reported in the paper.
* We observed a reduced accuracy when applying ms+flip augmentation to MsacleOCRNet during test time. This is probably due to that MscaleOCRNet has built internally multi-scale operations in the network.
