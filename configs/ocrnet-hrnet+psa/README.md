# PSA: Polarized Self-Attention: Towards High-quality Pixel-wise Regression

## Reference

> Huajun Liu, Fuqiang Liu, Xinyi Fan and Dong Huang. "Polarized Self-Attention: Towards High-quality Pixel-wise Regression." arXiv preprint arXiv:2107.00782v2 (2021).

## Performance

### Cityscapes

| Model            | Backbone    | Resolution | Training Iters | mIoU   | mIoU (flip) | mIoU (ms+flip) | Links           |
| ---------------- | ----------- | ---------- | -------------- | ------ | ----------- | -------------- | --------------- |
| OCRNet-HRNet+psa | HRNETV2_W48 | 1024x2048  | 80000          | 84.55% | 84.73%      | 84.92%         | model\|log\|vdl |

### Notes

Because the authors use **train + val set for training** and val set for evaluating, we only use the train set for training, the final accuracy is lower than that in paper.
