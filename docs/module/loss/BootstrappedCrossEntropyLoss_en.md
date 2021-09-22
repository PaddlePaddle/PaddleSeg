English | [简体中文](BootstrappedCrossEntropyLoss_cn.md)
## [BootstrappedCrossEntropyLoss](../../../paddleseg/models/losses/bootstrapped_cross_entropy.py)

Bootstrapped first uses samples to construct an initial classification, and then iteratively classifies unlabeled samples, and then uses the expanded training data to extract new seed rules for unlabeled samples.

[paper](https://arxiv.org/pdf/1412.6596.pdf)
```python
class paddleseg.models.losses.BootstrappedCrossEntropyLoss(
                        min_K, 
                        loss_th, 
                        weight = None, 
                        ignore_index = 255
)
```

## Bootstrapped cross entropy loss usage guidance

### Args
* **min_K**  (int): the minimum number of pixels to be counted in loss computation.
* **loss_th** (float): The loss threshold. Only loss that is larger than the threshold
            would be calculated.
* **weight** (tuple|list, optional): The weight for different classes. *Default:``None``*
* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
