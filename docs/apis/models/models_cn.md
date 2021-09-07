简体中文 | [English](models.md)
# paddleseg.models

该models子包中包含了以下21种用于图像语义分割的模型。

- [DeepLabV3+](#DeepLabV3)
- [DeepLabV3](#DeepLabV3-1)
- [FCN](#FCN)
- [OCRNet](#OCRNet)
- [PSPNet](#PSPNet)
- [ANN](#ANN)
- [BiSeNetV2](#BiSeNetV2)
- [DANet](#DANet)
- [FastSCNN](#FastSCNN)
- [GCNet](#GCNet)
- [GSCNN](#GSCNN)
- [HarDNet](#HarDNet)
- [UNet](#UNet)
- [U<sup>2</sup>Net](#U2Net)
- [U<sup>2</sup>Net+](#U2Net-1)
- [AttentionUNet](#AttentionUNet)
- [UNet++](#UNet-1)
- [DecoupledSegNet](#DecoupledSegNet)
- [ISANet](#ISANet)
- [EMANet](#EMANet)
- [DNLNet](#DNLNet)


## [DeepLabV3+](../../../paddleseg/models/deeplab.py)
```python
class paddleseg.models.DeepLabV3P(
        num_classes, 
        backbone, 
        backbone_indices = (0, 3), 
        aspp_ratios = (1, 6, 12, 18), 
        aspp_out_channels = 256, 
        align_corners = False, 
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 DeepLabV3Plus。

> 原文请参考：
[Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"](https://arxiv.org/abs/1802.02611)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (paddle.nn.Layer): 骨干网络，目前支持 Resnet50_vd/Resnet101_vd/Xception65。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。*默认:`` (0, 3)``*
* **aspp_ratios** (tuple, optional): ASSP module模块中使用的扩张率。
        如果 output_stride = 16, aspp_ratios 应该设定为 (1, 6, 12, 18)。
        如果 output_stride = 8, aspp_ratios 应该设定为 (1, 12, 24, 36)。
        *默认:``(1, 6, 12, 18)``*
* **aspp_out_channels** (int, optional): ASPP模块的输出通道数。*默认:``256``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [DeepLabV3](../../../paddleseg/models/deeplab.py)   
```python
class paddleseg.models.DeepLabV3(
        num_classes, 
        backbone, 
        backbone_indices = (3, ), 
        aspp_ratios = (1, 6, 12, 18), 
        aspp_out_channels = 256, 
        align_corners = False, 
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 DeepLabV3。

> 原文请参考：
[Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"](https://arxiv.org/pdf/1706.05587.pdf).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (paddle.nn.Layer): 骨干网络，目前支持 Resnet50_vd/Resnet101_vd/Xception65。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
       *默认:``(3, )``*
* **aspp_ratios** (tuple, optional): ASSP 模块中使用的扩张率。
        如果 output_stride = 16，aspp_ratios 应该被设置为 (1, 6, 12, 18)。
        如果 output_stride = 8，aspp_ratios 应该被设置为 (1, 12, 24, 36)。
        *默认:``(1, 6, 12, 18)``*
* **aspp_out_channels** (int, optional): ASPP模块的输出通道数。 *默认:``256``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [FCN](../../../paddleseg/models/deeplab.py)
```python
class paddleseg.models.FCN(
        num_classes,
        backbone_indices = (-1, ),
        backbone_channels = (270, ),
        channels = None
)
```

> 基于 PaddlePaddle 简单实现的 FCN。

> 原文请参考：
[Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"](https://arxiv.org/abs/1411.4038).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (paddle.nn.Layer): 骨干网络。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
        *默认:``(-1, )``*
* **channels** (int, optional): 在FCNHead的卷积层与最后一层之间的通道数。
        如果为None，它将是输入特征的通道数。*默认:``None``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*



## [OCRNet](../../../paddleseg/models/ocrnet.py)
```python
class paddleseg.models.OCRNet(
        num_classes,
        backbone,
        backbone_indices,
        ocr_mid_channels = 512,
        ocr_key_channels = 256,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 OCRNet。

> 原文请参考：
[Yuan, Yuhui, et al. "Object-Contextual Representations for Semantic Segmentation"](https://arxiv.org/pdf/1909.11065.pdf)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 骨干网络。
* **backbone_indices** (tuple): 该元组表示骨干网络输出的索引。该元组既可以含一个值，又可以是含个值。
            如果含两个值，则将第一个索引作为辅助层的深度监督特征；第二个索引将被视为像素表示的输入。如果只含一个值，该值将被应用于以上两种用途。 
* **ocr_mid_channels** (int, optional): OCRHead中middle channel的数目。*默认:``512``*
* **ocr_key_channels** (int, optional): ObjectAttentionBlock中 key channels 的数目。*默认:``256``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*



## [PSPNet](../../../paddleseg/models/pspnet.py)
```python
class paddleseg.models.PSPNet(
        num_classes,
        backbone,
        backbone_indices = (2, 3),
        pp_out_channels = 1024,
        bin_sizes = (1, 2, 3, 6),
        enable_auxiliary_loss = True,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 PSPNet。

> 原文请参考：
[Zhao, Hengshuang, et al. "Pyramid scene parsing network"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 骨干网络，目前支持 Resnet50/101。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
* **pp_out_channels** (int, optional): 在Pyramid Pooling模块后的 output channel 的数目。*默认:``1024``*
* **bin_sizes** (tuple, optional): 经池化后的特征图的输出尺寸。 *默认:``(1,2,3,6)``*
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。*默认:``True``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [ANN](../../../paddleseg/models/ann.py)
```python
class paddleseg.models.ANN(
        num_classes,
        backbone,
        backbone_indices = (2, 3),
        key_value_channels = 256,
        inter_channels = 512,
        psp_size = (1, 3, 6, 8),
        enable_auxiliary_loss = True,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 ANN 。

> 原文请参考：
[Zhen, Zhu, et al. "Asymmetric Non-local Neural Networks for Semantic Segmentation"](https://arxiv.org/pdf/1908.07678.pdf).

### Args
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 骨干网络，目前支持 Resnet50/101。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
* **key_value_channels** (int, optional): AFNB 和 APNB 模块中 self-attention 的键值通道映射。
            *默认:``256``*
* **inter_channels** (int, optional): APNB 模块的输入、输出通道数。*默认:``512``*
* **psp_size** (tuple, optional):  池化后的特征图大小。*默认:``(1, 3, 6, 8)``*
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。*默认:``True``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [BiSeNetV2](../../../paddleseg/models/bisenet.py)
```python
class paddleseg.models.BiSeNetV2(
        num_classes,
        lambd = 0.25,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 BiSeNet V2。

> 原文请参考：
[Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"](https://arxiv.org/abs/2004.02147)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **lambd** (float, optional): 控制语义分支通道大小的因素。*默认:``0.25``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [DANet](../../../paddleseg/models/danet.py)
```python
class paddleseg.models.DANet(
        num_classes,
        lambd = 0.25,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 DANet。

> 原文请参考：
[Fu, jun, et al. "Dual Attention Network for Scene Segmentation"](https://arxiv.org/pdf/1809.02983.pdf)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 骨干网络。
* **backbone_indices** (tuple): 元组中的两个值指示了骨干网络输出的索引。
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [FastSCNN](../../../paddleseg/models/fast_scnn.py)
```python
class paddleseg.models.FastSCNN(
        num_classes,
        enable_auxiliary_loss = True,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 FastSCNN。正如原论文中提到的，FastSCNN 是一种实时分割算法（123.5fps），即使对于高分辨率图像（1024x2048）也是如此。

> 原文请参考：
[Poudel, Rudra PK, et al. "Fast-scnn: Fast semantic segmentation network"](https://arxiv.org/pdf/1902.04502.pdf).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。
            如果为 True，辅助损失将被添加在 LearningToDownsample 模块后面。*默认:``False``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [GCNet](../../../paddleseg/models/gcnet.py)
```python
class paddleseg.models.GCNet(
        num_classes,
        backbone,
        backbone_indices = (2, 3),
        gc_channels = 512,
        ratio = 0.25,
        enable_auxiliary_loss = True,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 GCNet。

> 原文请参考：
[Cao, Yue, et al. "GCnet: Non-local networks meet squeeze-excitation networks and beyond"](https://arxiv.org/pdf/1904.11492.pdf).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 骨干网络，目前支持 Resnet50/101。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
* **gc_channels** (int, optional): 全局上下文块的输入通道数。*默认:``512``*
* **ratio** (float, optional): 表示注意力通道和 gc_channels 的比率。 *默认:``0.25``*
* **enable_auxiliary_loss** (bool, optional):一个 bool 值，指示是否添加辅助损失。*默认:``True``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [GSCNN](../../../paddleseg/models/gscnn.py)
```python
class paddleseg.models.GSCNN(
        num_classes,
        backbone,
        backbone_indices = (0, 1, 2, 3),
        aspp_ratios = (1, 6, 12, 18),
        aspp_out_channels = 256,
        align_corners = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 GSCNN。

> 原文请参考：
[Towaki Takikawa, et, al. "Gated-SCNN: Gated Shape CNNs for Semantic Segmentation"](https://arxiv.org/pdf/1907.05740.pdf)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (paddle.nn.Layer): 骨干网络，目前支持 Resnet50_vd/Resnet101_vd。
* **backbone_indices** (tuple, optional): 元组中的四个值指示了骨干网络输出的索引。
           *默认:``(0, 1, 2, 3)``*
* **aspp_ratios** (tuple, optional): ASSP module模块中使用的扩张率。
        如果 output_stride = 16, aspp_ratios 应该设定为 (1, 6, 12, 18)。
        如果 output_stride = 8, aspp_ratios 应该设定为 (1, 12, 24, 36)。
        *默认:``(1, 6, 12, 18)``*
* **aspp_out_channels** (int, optional): ASPP模块的输出通道数。*默认:``256``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [HarDNet](../../../paddleseg/models/hardnet.py)
```python
class paddleseg.models.HarDNet(
        num_classes,
        stem_channels = (16, 24, 32, 48),
        ch_list = (64, 96, 160, 224, 320),
        grmul = 1.7,
        gr = (10, 16, 18, 24, 32),
        n_layers = (4, 4, 8, 8, 8),
        align_corners = False,
        pretrained = None
)
```

> [实时] 基于 PaddlePaddle 实现的 FC-HardDNet 70。

> 原文请参考：
[Chao, Ping, et al. "HarDNet: A Low Memory Traffic Network"](https://arxiv.org/pdf/1909.00948.pdf)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **stem_channels** (tuple|list, optional): 进入编码器之前的通道数。 *默认:``(16, 24, 32, 48)``*
* **ch_list** (tuple|list, optional): 编码器中每个块的通道数。*默认:``(64, 96, 160, 224, 320)``*
* **grmul** (float, optional): HarDBlock 中的通道倍增因子，论文中用 m 表示。*默认:``1.7``*
* **gr** (tuple|list, optional): 每个HardBlock中的增长率，在论文中用 k 表示。 *默认:``(10, 16, 18, 24, 32)``*
* **n_layers** (tuple|list, optional): 每个HarDBlock中的层数。 *默认:``(4, 4, 8, 8, 8)``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [UNet](../../../paddleseg/models/unet.py)
```python
class paddleseg.models.UNet(
        num_classes,
        align_corners = False,
        use_deconv = False,
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 UNet。

> 原文请参考：
[Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **use_deconv** (bool, optional): bool 值指示是否在上采样中使用反卷积。
    如果为 False，则使用 resize_bilinear。 *默认:``False``*
* **pretrained** (str, optional): 用于模型微调的预训练模型的url或path。 *默认:``None``*

## [U<sup>2</sup>Net](../../../paddleseg/models/u2net.py)
```python
class paddleseg.models.U2Net(
        num_classes, 
        in_ch = 3, 
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 U^2-Net。

> 原文请参考：
[Xuebin Qin, et, al. "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"](https://arxiv.org/abs/2005.09007).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **in_ch** (int, optional): 输入通道数。 *默认:``3``*
* **pretrained** (str, optional): 用于模型微调的预训练模型的url或path。 *默认:``None``*

## [U<sup>2</sup>Net+](../../../paddleseg/models/u2net.py)
```python
class paddleseg.models.U2Netp(
        num_classes, 
        in_ch = 3, 
        pretrained = None
)
```
> 基于 PaddlePaddle 实现的 U^2-Netp。

> 原文请参考：
[Xuebin Qin, et, al. "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"](https://arxiv.org/abs/2005.09007).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **in_ch** (int, optional):  输入通道数。 *默认:``3``*
* **pretrained** (str, optional): 用于模型微调的预训练模型的url或path。 *默认:``None``*

## [AttentionUNet](../../../paddleseg/models/attention_unet.py)
```python
class paddleseg.models.AttentionUNet(num_classes, pretrained = None)
```

> 基于 PaddlePaddle 实现的 Attention-UNet。正如原论文中提到的，作者提出了一种新颖的注意力门 (AG)，它可以自动学习关注不同形状和大小的目标结构。使用 AG 训练的模型隐式地学习抑制输入图像中的不相关区域，同时突出显示对特定任务有用的显著特征。

> 原文请参考：
[Oktay, O, et, al. "Attention u-net: Learning where to look for the pancreas."](https://arxiv.org/pdf/1804.03999.pdf).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [UNet++](../../../paddleseg/models/unet_plusplus.py)
```python
class UNetPlusPlus(
    in_channels,
    num_classes,
    use_deconv = False,
    align_corners = False,
    pretrained = None,
    is_ds = True
)
```

> 基于 PaddlePaddle 实现的 UNet++。

> 原文请参考：
[Zongwei Zhou, et, al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"](https://arxiv.org/abs/1807.10165).

### 参数
* **in_channels** (int): 输入图像的通道数。
* **num_classes** (int): 相互独立的目标类别的数量。
* **use_deconv** (bool, optional): bool 值指示是否在上采样中使用反卷积。
    如果为 False，则使用 resize_bilinear。 *默认:``False``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 用于模型微调的预训练模型的url或path。 *默认:``None``*
* **is_ds** (bool): 是否使用deep supervision。 *默认:``True``*

## [DecoupledSegNet](../../../paddleseg/models/decoupled_segnet.py)
```python
class DecoupledSegNet(
    num_classes,
    backbone,
    backbone_indices = (0, 3),
    aspp_ratios = (1, 6, 12, 18),
    aspp_out_channels = 256,
    align_corners = False,
    pretrained = None
)
```

> 基于 PaddlePaddle 实现的 DecoupledSegNet。

> 原文请参考：
[Xiangtai Li, et, al. "Improving Semantic Segmentation via Decoupled Body and Edge Supervision"](https://arxiv.org/pdf/2007.10035.pdf)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (paddle.nn.Layer): 骨干网络，目前支持 Resnet50_vd/Resnet101_vd。
* **backbone_indices** (tuple, optional): 元组中的两个值指示了骨干网络输出的索引。
           *默认:``(0, 3)``*
* **aspp_ratios** (tuple, optional): ASSP module模块中使用的扩张率。
        如果 output_stride = 16, aspp_ratios 应该设定为 (1, 6, 12, 18)。
        如果 output_stride = 8, aspp_ratios 应该设定为 (1, 12, 24, 36)。
        *默认:``(1, 6, 12, 18)``*
* **aspp_out_channels** (int, optional): ASPP模块的输出通道数。*默认:``256``*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [ISANet](../../../paddleseg/models/isanet.py)
```python
class paddleseg.models.ISANet(
        num_classes, 
        backbone, 
        backbone_indices = (2, 3), 
        isa_channels = 256, 
        down_factor = (8, 8), 
        enable_auxiliary_loss = True, 
        align_corners = False, 
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 ISANet。

> 原文请参考：
[Lang Huang, et al. "Interlaced Sparse Self-Attention for Semantic Segmentation"](https://arxiv.org/abs/1907.12273).

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 一个骨干网络。
* **backbone_indices** (tuple): 元组中的两个值指示了骨干网络输出的索引。
* **isa_channels** (int): ISA 模块的通道数。
* **down_factor** (tuple): 将高度和宽度尺寸划分为 (Ph, PW) 组。
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。*默认:``True``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*


## [EMANet](../../../paddleseg/models/emanet.py)
```python
class paddleseg.models.EMANet(
        num_classes, 
        backbone, 
        backbone_indices = (2, 3), 
        ema_channels = 512, 
        gc_channels = 256, 
        num_bases = 64, 
        stage_num = 3, 
        momentum = 0.1, 
        concat_input = True, 
        enable_auxiliary_loss = True, 
        align_corners = False, 
        pretrained = None
)
```

> 基于 PaddlePaddle 实现的 EMANet。

> 原文请参考：
[Xia Li, et al. "Expectation-Maximization Attention Networks for Semantic Segmentation"](https://arxiv.org/abs/1907.13426)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 一个骨干网络。
* **backbone_indices** (tuple): 元组中的两个值指示了骨干网络输出的索引。
* **ema_channels** (int): EMA 模块的通道数。
* **gc_channels** (int): 全局上下文块的输入通道。
* **num_bases** (int): base的数目。
* **stage_num** (int): EM的迭代次数。
* **momentum** (float): 更新base的参数。
* **concat_input** (bool): 是否在分类层之前连接卷积层的输入和输出。 *默认:``True``*
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。 *默认:``True``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*

## [DNLNet](../../../paddleseg/models/dnlnet.py)
```python
class paddleseg.models.DNLNet(
    num_classes, backbone, 
    backbone_indices = (2, 3), 
    reduction = 2, 
    use_scale = True, 
    mode = 'embedded_gaussian', 
    temperature = 0.05, 
    concat_input = True, 
    enable_auxiliary_loss = True, 
    align_corners = False, 
    pretrained = None
)
```

> 基于 PaddlePaddle 实现的 DNLNet。

> 原文请参考：
[Minghao Yin, et al. "Disentangled Non-Local Neural Networks"](https://arxiv.org/abs/2006.06668)

### 参数
* **num_classes** (int): 相互独立的目标类别的数量。
* **backbone** (Paddle.nn.Layer): 一个骨干网络。
* **backbone_indices** (tuple): 元组中的两个值指示了骨干网络输出的索引。
* **reduction** (int): 投影变换的缩减因子。*默认:``2``*
* **use_scale** (bool): 是否按 sqrt(1/inter_channels) 缩放pairwise_weight。 *默认:``False``*
* **mode** (str): nonlocal 模式。可选项有'embedded_gaussian',
            'dot_product'。*默认:``'embedded_gaussian'``*
* **temperature** (float): 使用温度调节注意力。*默认:``0.05``*
* **concat_input** (bool): 是否在分类层之前连接卷积层的输入和输出。*默认:``True``*
* **enable_auxiliary_loss** (bool, optional): 一个 bool 值，指示是否添加辅助损失。 *默认:``True``*
* **align_corners** (bool): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。*默认:``False``*
* **pretrained** (str, optional): 预训练模型的url或path。 *默认:``None``*
