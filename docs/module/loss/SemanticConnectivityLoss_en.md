English | [简体中文](SemanticConnectivityLoss_cn.md)
## [SemanticConnectivityLoss](../../../paddleseg/models/losses/semantic_connectivity_loss.py)
SCL (Semantic Connectivity-aware Learning) framework, which introduces a SC Loss (Semantic Connectivity-aware Loss)
to improve the quality of segmentation results from the perspective of connectivity. Support multi-class segmentation.

The original article refers to
    Lutao Chu, Yi Liu, Zewu Wu, Shiyu Tang, Guowei Chen, Yuying Hao, Juncai Peng, Zhiliang Yu, Zeyu Chen, Baohua Lai, Haoyi Xiong.
    "PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset"
    In WACV 2022 workshop
    https://arxiv.org/abs/2112.07146

Running process:
Step 1. Connected Components Calculation
Step 2. Connected Components Matching and SC Loss Calculation

```python
class paddleseg.models.losses.SemanticConnectivityLoss(
            ignore_index = 255,
            max_pred_num_conn = 10,
            use_argmax = True
)
```

## Semantic Connectivity Learning usage guidance

### Args
* **ignore_index** (int): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **max_pred_num_conn** (int): Maximum number of predicted connected components. At the beginning of training,
                there will be a large number of connected components, and the calculation is very time-consuming.
                Therefore, it is necessary to limit the maximum number of predicted connected components,
                and the rest will not participate in the calculation.
* **use_argmax** (bool): Whether to use argmax for logits.
