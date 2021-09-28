English| [简体中文](LovaszHingeLoss_cn.md)
## [LovaszHingeLoss](../../../paddleseg/models/losses/lovasz_loss.py)

Hinge Loss is a loss function improved on the simple step loss function that is not continuous and smooth. For positive samples, the output of Hinge Loss should be greater than or equal to 1; for positive samples, the output of Hinge Loss should be less than or equal to -1.

```python
class paddleseg.models.losses.LovaszHingeLoss(ignore_index = 255)
```

## Lovasz hinge loss usage guidance

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
