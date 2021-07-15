English | [简体中文](RelaxBoundaryLoss_cn.md)
## [RelaxBoundaryLoss](../../../paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py)

In order to improve the segmentation effect, the processing of the boundary is essential. Usually a neural network is used to predict the boundary map. In this training process, the boundary is usually used as the dividing basis, and the pixels sensitive to the boundary are divided into several categories, and finally the cross entropy is calculated and output as a loss function.

```python

class paddleseg.models.losses.RelaxBoundaryLoss(
                border = 1,
                calculate_weights = False,
                upper_bound = 1.0,
                ignore_index = 255
)
```

## Relax boundary loss usage guidance

### Args
* **border**  (int, optional): The value of border to relax. *Default:`` 1``*
* **calculate_weights** (bool, optional): Whether to calculate weights for every classes. *Default:``False``*
* **upper_bound** (float, optional): The upper bound of weights if calculating weights for every classes. *Default:``1.0``*
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*