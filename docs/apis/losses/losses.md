English | [简体中文](losses_cn.md)
# [paddleseg.models.losses](../../../paddleseg/models/losses)


## [BCELoss](../../../paddleseg/models/losses/binary_cross_entropy_loss.py)
```python
class paddleseg.models.losses.BCELoss(
            weight = None,
            pos_weight = None,
            ignore_index = 255,
            edge_label = False
)
```

> The implement of binary cross entropy loss.

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



## [BootstrappedCrossEntropyLoss](../../../paddleseg/models/losses/bootstrapped_cross_entropy.py)
```python
class paddleseg.models.losses.BootstrappedCrossEntropyLoss(
                        min_K, 
                        loss_th, 
                        weight = None, 
                        ignore_index = 255
)
```
> The implement of bootstrapped cross entropy loss.

### Args
* **min_K**  (int): the minimum number of pixels to be counted in loss computation.
* **loss_th** (float): The loss threshold. Only loss that is larger than the threshold
            would be calculated.
* **weight** (tuple|list, optional): The weight for different classes. *Default:``None``*
* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*




## [CrossEntropyLoss](../../../paddleseg/models/losses/cross_entropy_loss.py)
```python
class paddleseg.models.losses.CrossEntropyLoss(
                weight = None, 
                ignore_index = 255, 
                top_k_percent_pixels = 1.0
)
```

> The implement of cross entropy loss.

### Args
* **weight**  (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.The weights of various types can be adjusted under conditions such as unbalanced samples of multiple types.
            *Default ``None``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **top_k_percent_pixels** (float, optional): The value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.



## [RelaxBoundaryLoss](../../../paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py)
```python
class paddleseg.models.losses.RelaxBoundaryLoss(
                border = 1,
                calculate_weights = False,
                upper_bound = 1.0,
                ignore_index = 255
)
```

> The implement of relax boundary loss.

### Args
* **border**  (int, optional): The value of border to relax. *Default:`` 1``*
* **calculate_weights** (bool, optional): Whether to calculate weights for every classes. *Default:``False``*
* **upper_bound** (float, optional): The upper bound of weights if calculating weights for every classes. *Default:``1.0``*
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*




## [DiceLoss](../../../paddleseg/models/losses/dice_loss.py)
```python
class paddleseg.models.losses.DiceLoss(
            ignore_index = 255, 
            smooth = 0.
)
```

> The implement of dice loss.

### Args
* **ignore_index**  (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **smooth** (float, optional): The smooth parameter can be added to prevent the division by 0 exception. You can also set a larger smoothing value (Laplacian smoothing) to avoid overfitting.*Default:``0``*




## [EdgeAttentionLoss](../../../paddleseg/models/losses/edge_attention_loss.py)
```python
class paddleseg.models.losses.EdgeAttentionLoss(
                edge_threshold = 0.8, 
                ignore_index = 255
)
```

> The implement of edge attention loss.

### Args
* **edge_threshold** (float): The pixels whose values are greater than edge_threshold are treated as edges.
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*



## [DualTaskLoss](../../../paddleseg/models/losses/gscnn_dual_task_loss.py)
```python
class paddleseg.models.losses.DualTaskLoss(
            ignore_index = 255, 
            tau = 0.5
)
```

> The implement of dual task loss.

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **tau** (float): the tau of gumbel softmax sample.



## [L1Loss](../../../paddleseg/models/losses/l1_loss.py)
```python
class paddleseg.models.losses.L1Loss(
            reduction = 'mean', 
            ignore_index = 255
)
```

> The implement of L1 loss.

### Args
* **reduction** (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.

    > - If `reduction` is ``'none'``, the unreduced loss is returned.
    > - If `reduction` is ``'mean'``, the reduced mean loss is returned.
    > - If `reduction` is ``'sum'``, the reduced sum loss is returned.
    > - *Default:``'mean'``*
* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*




## [MSELoss](../../../paddleseg/models/losses/mean_square_error_loss.py)
```python
class paddleseg.models.losses.MSELoss(
            reduction = 'mean', 
            ignore_index = 255
)
```

> The implement of mean square error loss.

### Args
* **reduction** (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.

    > - If `reduction` is ``'none'``, the unreduced loss is returned.
    > - If `reduction` is ``'mean'``, the reduced mean loss is returned.
    > - If `reduction` is ``'sum'``, the reduced sum loss is returned.
    > - *Default:``'mean'``*

* **ignore_index** (int, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*




## [OhemCrossEntropyLoss](../../../paddleseg/models/losses/ohem_cross_entropy_loss.py)
```python
class paddleseg.models.losses.OhemCrossEntropyLoss(
                thresh = 0.7, 
                min_kept = 10000, 
                ignore_index = 255
)
```

> The implement of ohem cross entropy loss.

### Args
* **thresh** (float, optional): The threshold of ohem. *Default:``0.7``*
* **min_kept** (int, optional): Specify the minimum number of pixels to keep for calculating the loss function.``min_kept`` is used in conjunction with ``thresh``: If ``thresh`` is set too high, it may result in no input value to the loss function in this round of iteration, so setting this value can ensure that at least the top ``min_kept`` elements will not be filtered out. *Default:``10000``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*



## [OhemEdgeAttentionLoss](../../../paddleseg/models/losses/ohem_edge_attention_loss.py)
```python
class paddleseg.models.losses.OhemEdgeAttentionLoss(
                edge_threshold = 0.8,
                thresh = 0.7,
                min_kept = 5000,
                ignore_index = 255
)
```

> The implement of ohem edge attention loss.

### Args
* **edge_threshold** (float, optional): The pixels greater edge_threshold as edges. *Default:`` 0.8``*
* **thresh** (float, optional): The threshold of ohem. *Default:`` 0.7``*
* **min_kept** (int, optional): Specify the minimum number of pixels to keep for calculating the loss function.``min_kept`` is used in conjunction with ``thresh``: If ``thresh`` is set too high, it may result in no input value to the loss function in this round of iteration, so setting this value can ensure that at least the top ``min_kept`` elements will not be filtered out. *Default:``5000``*
* **ignore_index** (int64, optional): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*


## [LovaszSoftmaxLoss](../../../paddleseg/models/losses/lovasz_loss.py)
```python
class paddleseg.models.losses.LovaszSoftmaxLoss(
            ignore_index = 255, 
            classes = 'present'
)
```

> The implement of multi-class Lovasz-Softmax loss.

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*
* **classes** (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.


## [LovaszHingeLoss](../../../paddleseg/models/losses/lovasz_loss.py)
```python
class paddleseg.models.losses.LovaszHingeLoss(ignore_index = 255)
```

> The implement of binary Lovasz hinge loss.

### Args
* **ignore_index** (int64): Specify a pixel value to be ignored in the annotated image
            and does not contribute to the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding to the original image will not be used as the independent variable of the loss function. *Default:``255``*


## [MixedLoss](../../../paddleseg/models/losses/mixed_loss.py)
```python
class paddleseg.models.losses.MixedLoss(losses, coef)
```

> Weighted computations for multiple Loss.
> The advantage is that mixed loss training can be achieved without changing the networking code.

### Args
* **losses** (list of nn.Layer): A list consisting of multiple loss classes
* **coef** (float|int): Weighting coefficient of multiple loss

### Returns
* A callable object of MixedLoss.
