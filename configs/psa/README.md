# PSA: Polarized Self-Attention: Towards High-quality Pixel-wise Regression

## Reference

> Huajun Liu, Fuqiang Liu, Xinyi Fan and Dong Huang. "Polarized Self-Attention: Towards High-quality Pixel-wise Regression." arXiv preprint arXiv:2107.00782v2 (2021).

## Performance

### Cityscapes

| Model            | Backbone    | Resolution | Training Iters | mIoU   | mIoU (flip) | mIoU (ms+flip) | Links           |
| ---------------- | ----------- | ---------- | -------------- | ------ | ----------- | -------------- | --------------- |
| OCRNet-HRNet+psa | HRNETV2_W48 | 1024x2048  | 80000          | 84.55% | 84.73%      | 84.92%         | model\|log\|vdl |

### Notes

Since we cannot reproduce the training results from [the authors' official repo](https://github.com/DeLightCMU/PSA), we follow the settings in the original paper to train and evaluate our models, and the final accuracy is lower than that reported in the paper.
