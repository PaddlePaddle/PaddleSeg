English | [简体中文](MSELoss_en.md)
## [MSELoss](../../../paddleseg/models/losses/mean_square_error_loss.py)
The root mean square refers to the average of the square of the distance between the model's predicted value and the sample's true value.

```python
class paddleseg.models.losses.MSELoss(
            reduction = 'mean',
            ignore_index = 255
)
```

## Mean square error loss usage guidances

### Args
* **reduction** (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.

    > - If `reduction` is ``'none'``, the unreduced loss is returned.
    > - If `reduction` is ``'mean'``, the reduced mean loss is returned.
    > - If `reduction` is ``'sum'``, the reduced sum loss is returned.
    > - *Default:``'mean'``*

* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
