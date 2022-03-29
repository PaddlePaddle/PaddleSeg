English | [简体中文](LovaszSoftmaxLoss_cn.md)
## [LovaszSoftmaxLoss](../../../paddleseg/models/losses/lovasz_loss.py)

Lovasz softmax loss is suitable for multi-classification problems. The work was published on CVPR 2018.
[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html)

```python
class paddleseg.models.losses.LovaszSoftmaxLoss(
            ignore_index = 255,
            classes = 'present'
)
```

## Lovasz-Softmax loss usage guidance

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **classes** (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.
