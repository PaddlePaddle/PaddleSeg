English | [简体中文](DiceLoss_cn.md)
## [DiceLoss](../../../paddleseg/models/losses/dice_loss.py)

Dice Loss is a loss function widely used in medical image segmentation tasks. The Dice coefficient is a function used to measure the degree of similarity between sets. In the semantic segmentation task, we can understand the degree of similarity between the current model and the real model in the real world. The calculation process of the Dice loss function includes the dot multiplication between the predicted segmentation map and the GT segmentation map, the cumulative sum of each position of the dot multiplication result, and finally the calculation of the value of 1-Dice as the output of the loss function, that is, Dice = 1-2(|X∩Y|/|X|+|Y|). You can use the Laplacian smoothing coefficient. After adding the coefficient to the numerator and denominator, you can avoid the division by 0 exception and reduce overfitting. That is, Dice_smooth = 1-2((|X∩Y|+smooth) / (|X|+|Y|+smooth))

```python。

class paddleseg.models.losses.DiceLoss(
            ignore_index = 255, 
            smooth = 0.
)
```

## Dice loss usage guidance

### Args
* **ignore_index**  (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **smooth** (float, optional): The smooth parameter can be added to prevent the division by 0 exception. You can also set a larger smoothing value (Laplacian smoothing) to avoid overfitting.*Default:``0``*