English | [简体中文](RelaxBoundaryLoss_cn.md)
## [RelaxBoundaryLoss](../../../paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py)

Relax boundary loss is composed of multiple parts: the loss of main features, the loss of edge features, and the cross-entropy loss. RelaxBoundaryLoss is a loss function designed for DecoupleSegNet. This model makes predictions for the categories (≥2) that boundary pixels may belong to. This loss is to maximize the sum of the probabilities of each category of a single pixel under the constraint of boundary relaxation.

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
