English | [简体中文](CrossEntropyLoss_cn.md)
## [CrossEntropyLoss](../../../paddleseg/models/losses/cross_entropy_loss.py)

The cross entropy (CE) method has become one of the most popular loss functions due to its simplicity and effectiveness, allowing adjustment of the weights of different categories of pixels. In many semantic segmentation tasks, cross entropy relies on enough objective function calls to accurately estimate the best parameters of the underlying distribution.
CrossEntropyLoss is often used for multi-pixel segmentation tasks. It describes the difference between two probability distributions. It can be used to describe the gap between the current model and the actual model (during the training process, we temporarily think that the given label set It is the model in the real world). Note: Logistic regression in machine learning algorithms is a special case of this kind of cross-entropy.
```python
class paddleseg.models.losses.CrossEntropyLoss(
                weight = None, 
                ignore_index = 255, 
                top_k_percent_pixels = 1.0
)
```

## Cross entropy loss usage guidance

### Args
* **weight**  (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.The weights of various types can be adjusted under conditions such as unbalanced samples of multiple types.
            *Default ``None``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **top_k_percent_pixels** (float, optional): The value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.