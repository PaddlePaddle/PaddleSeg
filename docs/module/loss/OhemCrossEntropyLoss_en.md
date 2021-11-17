English | [简体中文](OhemCrossEntropyLoss_cn.md)
## [OhemCrossEntropyLoss](../../../paddleseg/models/losses/ohem_cross_entropy_loss.py)

OHEM aims to handling difficult samples. In some cases, there are unbalanced classes, and labelling all pixels is difficult or even impossible, which will severely restrict the performance of the model. The OHEM algorithm will distinguish difficult samples based on the loss of the samples input to the model. These difficult samples have poor classification accuracy and will produce greater losses.

```python
class paddleseg.models.losses.OhemCrossEntropyLoss(
                thresh = 0.7,
                min_kept = 10000,
                ignore_index = 255
)
```

## Ohem cross entropy loss usage guidance

### Args
* **thresh** (float, optional): The threshold of ohem. *Default:``0.7``*
* **min_kept** (int, optional): Specify the minimum number of pixels to keep for calculating the loss function.``min_kept`` is used in conjunction with ``thresh``: If ``thresh`` is set too high, it may result in no input value to the loss function in this round of iteration, so setting this value can ensure that at least the top ``min_kept`` elements will not be filtered out. *Default:``10000``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
