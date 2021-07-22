English | [简体中文](EdgeAttentionLoss_cn.md)
## [EdgeAttentionLoss](../../../paddleseg/models/losses/edge_attention_loss.py)
It is suitable for multi-task training scenarios where the encoder extracts the edge and the decoder performs weighted aggregation. It is a method of combining edge detection and attention mechanism for multi-loss combined output.

```python
class paddleseg.models.losses.EdgeAttentionLoss(
                edge_threshold = 0.8, 
                ignore_index = 255
)
```

## Edge attention loss usage guidance

### Args
* **edge_threshold** (float): The pixels whose values are greater than edge_threshold are treated as edges.
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*