English | [简体中文](BCELoss_cn.md)
# [BCELoss](../../../paddleseg/models/losses/binary_cross_entropy_loss.py)


Binary cross entropy is suitable for handling binary classification and multi-label classification tasks.The binary cross entropy is based on the probability model of the annotated map, and the binary semantic segmentation model is used to calculate the KL divergence. According to the Gibbs inequality, the cross entropy of the two is greater than the entropy of the semantic segmentation probability model. When calculating BCELoss, we usually ignore the entropy of the semantic segmentation probability model (because it is a constant), and only use a part of the KL divergence as the loss function.


```python
class paddleseg.models.losses.BCELoss(
            weight = None,
            pos_weight = None,
            ignore_index = 255,
            edge_label = False
)
```

## BCELoss usage guidance


### Args
* **weight**  (Tensor | str, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,the data type is float32, float64.
            If type is str, it should equal to 'dynamic'.
            It will compute weight dynamically in every step.
            *Default:``'None'``*
* **pos_weight** (float|str, optional): A weight of positive examples. If type is str,
            it should equal to 'dynamic'. It will compute weight dynamically in every step.
            *Default:``'None'``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient. When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **edge_label** (bool, optional): Whether to use edge label. *Default:``False``*
