# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

## Reference
> Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zheng-Ning Liu, Ming-Ming Cheng and Shi-Min Hu. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
>
> arXiv preprint arXiv:2207.13600 (2022).

## Performance

### Cityscapes

|  Model  | Backbone | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |             Links               |
| :-----: | :------: | :--------: | :------------: | :----: | :---------: | :------------: | :-----------------------------: |
| SegNeXt | MSCAN_T  | 1024x1024  |     160000     | 80.25% |   80.58%    |     80.75%     | [model]() \| [log]() \| [vdl]() |

**Note: There are some potential errors that will cause training collapse when use multiple GPUs. Or you can train SegNeXt without customing learning rate and weight decay of each parameters. You also need to expand learning rate by 10 times as the same time. The result is as following.**

|  Model  | Backbone | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |              Links              |
| :-----: | :------: | :--------: | :------------: | :----: | :---------: | :------------: | :---------------------------:   |
| SegNeXt | MSCAN_T  | 1024x1024  |     160000     | 81.04% |   81.20%    |     81.43%     | [model]() \|[log]() \| [vdl]()  |
