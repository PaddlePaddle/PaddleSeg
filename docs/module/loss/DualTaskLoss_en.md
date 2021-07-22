English | [简体中文](DualTaskLoss_cn.md)
## [DualTaskLoss](../../../paddleseg/models/losses/gscnn_dual_task_loss.py)
Dual-task consistency used for semi-supervised learning to constrain the model. DualTaskLoss aims to strengthen the consistency between multiple tasks.

```python
class paddleseg.models.losses.DualTaskLoss(
            ignore_index = 255, 
            tau = 0.5
)
```

## Dual task  loss usage guidance

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **tau** (float): the tau of gumbel softmax sample.