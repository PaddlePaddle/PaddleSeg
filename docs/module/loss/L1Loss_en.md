English | [简体中文](L1Loss_cn.md)
## [L1Loss](../../../paddleseg/models/losses/l1_loss.py)
Take the distance between the output predicted value and the true label value as the output of the loss function, and you can choose to use the 'reduction' strategy to perform certain processing on the direct calculation result of the loss.

```python
class paddleseg.models.losses.L1Loss(
            reduction = 'mean', 
            ignore_index = 255
)
```

## L1 loss usage guidance

### Args
* **reduction** (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.

    > - If `reduction` is ``'none'``, the unreduced loss is returned.
    > - If `reduction` is ``'mean'``, the reduced mean loss is returned.
    > - If `reduction` is ``'sum'``, the reduced sum loss is returned.
    > - *Default:``'mean'``*
* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*