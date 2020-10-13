
# API References
-----

```python
class paddleseg.models.OCRNet(
    num_classes,
    backbone,
    backbone_indices=None,
    ocr_mid_channels=512,
    ocr_key_channels=256,
    pretrained=None
)
```

The OCRNet implementation based on PaddlePaddle. The original article refers to [Yuan, Yuhui, et al. "Object-Contextual Representations for Semantic Segmentation"](https://arxiv.org/pdf/1909.11065.pdf)

> **Parameters**
>
> *num_classes(int)*: the unique number of target classes.
>
> *backbone(Paddle.nn.Layer)*: backbone network.
>
> *backbone_indices(tuple)*: two values in the tuple indicate the indices of output of backbone. the first index will be taken as a deep-supervision feature in auxiliary layer; the second one will be taken as input of pixel representation.
>
> *ocr_mid_channels(int)*: the number of middle channels in OCRHead.
>
> *ocr_key_channels(int)*: the number of key channels in ObjectAttentionBlock.
>
> *pretrained(str)*: the path or url of pretrained model. Default to None.

-----

```python
class paddleseg.models.BiSeNetv2(
    num_classes,
    backbone,
    backbone_indices=None,
    ocr_mid_channels=512,
    ocr_key_channels=256,
    pretrained=None
)
```

The BiSeNet V2 implementation based on PaddlePaddle. The original article refers to [Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"](https://arxiv.org/abs/2004.02147)

> **Parameters**
>
> *num_classes(int)*: the unique number of target classes.
>
> *lambd(float)*: factor for controlling the size of semantic branch channels. Default to 0.25.
>
> *pretrained(str)*: the path or url of pretrained model. Default to None.
