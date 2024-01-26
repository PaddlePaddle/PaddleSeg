English | [简体中文](OhemEdgeAttentionLoss_cn.md)
## [OhemEdgeAttentionLoss](../../../paddleseg/models/losses/ohem_edge_attention_loss.py)
The OHEM algorithm will distinguish difficult samples based on the loss of the samples input to the model. These difficult samples have poor classification accuracy and will produce greater losses. In the above cases, if you want to improve the performance of edge extraction, you can use this loss function.
```python
class paddleseg.models.losses.OhemEdgeAttentionLoss(
                edge_threshold = 0.8,
                thresh = 0.7,
                min_kept = 5000,
                ignore_index = 255
)
```

## Ohem edge attention loss usage guidance

### Args
* **edge_threshold** (float, optional): The pixels greater edge_threshold as edges. *Default:`` 0.8``*
* **thresh** (float, optional): The threshold of ohem. *Default:`` 0.7``*
* **min_kept** (int, optional): Specify the minimum number of pixels to keep for calculating the loss function.``min_kept`` is used in conjunction with ``thresh``: If ``thresh`` is set too high, it may result in no input value to the loss function in this round of iteration, so setting this value can ensure that at least the top ``min_kept`` elements will not be filtered out. *Default:``5000``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
