简体中文 | [English](backbones.md)
# paddleseg.models.backbone

该models子包中包含了为语义分割模型提取特征的骨干网络。

- [ResNet_vd](#ResNet_vd)
- [HRNet](#HRNet)
- [MobileNetV3](#MobileNetV3)
- [XceptionDeeplab](#xceptiondeeplab)


## [ResNet_vd](../../../paddleseg/models/backbones/resnet_vd.py)
ResNet_vd 骨干网络源于["Bag of Tricks for Image Classification with Convolutional Neural Networks"](https://arxiv.org/pdf/1812.01187.pdf)
```python
class paddleseg.models.backbones.Resnet_vd(
            layers = 50, 
            output_stride = None, 
            multi_grid = (1, 1, 1), 
            lr_mult_list = (0.1, 0.1, 0.2, 0.2), 
            pretrained = None
)
```

### 参数
* **layers** (int, optional): ResNet_vd的层数。 支持的层数有[18, 34, 50, 101, 152, 200]。 *默认: 50*
* **output_stride** (int, optional): 与输入图像相比，输出特征的缩放步长，该参数将影响下采样倍数。该参数应为 8 或 16。 *默认: 8*
* **multi_grid** (tuple|list, optional): stage 4的grid设置，用以扩大卷积的感受野。 *默认: (1, 1, 1)*
* **pretrained** (str, optional): 预训练模型的路径。
```python
paddleseg.models.backbones.ResNet18_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 18 。

```python
paddleseg.models.backbones.ResNet34_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 34 。

```python
paddleseg.models.backbones.ResNet50_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 50 。

```python
paddleseg.models.backbones.ResNet101_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 101 。

```python
paddleseg.models.backbones.ResNet152_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 152 。

```python
padddelseg.models.backbones.ResNet200_vd(**args)
```
> 返回 ResNet_vd 类的一个对象，其层数为 200 。

## [HRNet](../../../paddleseg/models/backbones/hrnet.py)
HRNet 骨干网络源于 ["HRNet：Deep High-Resolution Representation Learning for Visual Recognition"](https://arxiv.org/pdf/1908.07919.pdf)
```python
class paddleseg.models.backbones.HRNet(
            pretrained = None, 
            stage1_num_modules = 1, 
            stage1_num_blocks = (4,), 
            stage1_num_channels = (64,), 
            stage2_num_modules = 1, 
            stage2_num_blocks = (4, 4), 
            stage2_num_channels = (18, 36), 
            stage3_num_modules = 4, 
            stage3_num_blocks = (4, 4, 4), 
            stage3_num_channels = (18, 36, 72), 
            stage4_num_modules = 3, 
            stage4_num_blocks = (4, 4, 4, 4), 
            stage4_num_channels = (18, 36, 72, 14), 
            has_se = False, 
            align_corners = False
)
```

### 参数
* **pretrained** (str, optional): 预训练模型的路径。
* **stage1_num_modules** (int, optional): stage1 的模块数量。 *默认: 1*
* **stage1_num_blocks** (list, optional): stage1 的每个模块的block数。 *默认: (4,)*
* **stage1_num_channels** (list, optional): stage1 的每个分支的通道数。 *默认: (64,)*
* **stage2_num_modules** (int, optional): stage2 的模块数量。*默认: 1。
* **stage2_num_blocks** (list, optional): stage2 的每个模块的block数。 *默认: (4, 4)*
* **stage2_num_channels** (list, optional): stage2 的每个分支的通道数。 *默认: (18, 36)*
* **stage3_num_modules** (int, optional): stage3 的模块数量。 *默认: 4*
* **stage3_num_blocks** (list, optional): stage3 的每个模块的block数。 *默认: (4, 4, 4)*
* **stage3_num_channels** (list, optional): stage3 的每个分支的通道数。 *默认: (18, 36, 72)*
* **stage4_num_modules** (int, optional): stage4 的模块数量。*默认: 3*
* **stage4_num_blocks** (list, optional): stage4 的每个模块的block数。*默认: (4, 4, 4, 4)*
* **stage4_num_channels** (list, optional): stage4 的每个分支的通道数。 *默认: (18, 36, 72, 144)*
* **has_se** (bool, optional): 是否使用Squeeze-and-Excitation 模块。 *默认: False*
* **align_corners** (bool, optional): F.interpolate 的一个参数。当特征大小为偶数时应设置为 False，例如 1024x512；
                否则为 True, 例如 769x769。 *默认:: False*
```python
paddleseg.models.backbones.HRNet_W18_Small_V1(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 18 且小于 HRNet_W18_Small_V2。

```python
paddleseg.models.backbones.HRNet_W18_Small_V2(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 18 且小于 HRNet_W18。

```python
paddleseg.models.backbones.HRNet_W18(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 18。

```python
paddleseg.models.backbones.HRNet_W30(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 30。

```python
paddleseg.models.backbones.HRNet_W32(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 32。

```python
paddleseg.models.backbones.HRNet_W40(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 40。

```python
paddleseg.models.backbones.HRNet_W44(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 44。

```python
paddleseg.models.backbones.HRNet_W48(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 48。

```python
paddleseg.models.backbones.HRNet_W60(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 60。

```python
paddleseg.models.backbones.HRNet_W64(**kwargs)
```
> 返回 HRNet 类的一个对象，其宽度为 64。



## [MobileNetV3](../../../paddleseg/models/backbones/mobilenetv3.py)
MobileNetV3 骨干网络源于 ["Searching for MobileNetV3"](https://arxiv.org/pdf/1905.02244.pdf).
```python
class paddleseg.models.backbones.MobileNetV3(
            pretrained = None, 
            scale = 1.0, 
            model_name = "small",
            output_stride = None
)
```

### 参数
* **pretrained** (str, optional): 预训练模型的路径。
* **scale** (float, optional): 通道调整的尺度。建议：相对small模型而言，对large模型设置更高的scale。 *默认: 1.0*
* **model_name** (str, optional): 模型名称。它决定了MobileNetV3的类型。该参数应为 'small' 或 'large'之一。 *默认: 'small'*
* **output_stride** (int, optional): 与输入图像相比，输出特征的步长。 该参数应为 [2, 4, 8, 16, 32]中之一。 *默认: None*

```python
paddleseg.models.backbones.MobileNetV3_small_x0_35(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.35，且 model_name为 'small'。

```python
paddleseg.models.backbones.MobileNetV3_small_x0_5(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.5，且 model_name为 'small'。

```python
paddleseg.models.backbones.MobileNetV3_small_x0_75(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.75，且 model_name为 'small'。

```python
paddleseg.models.backbones.MobileNetV3_small_x1_0(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 1.0，且 model_name为 'small'。

```python
paddleseg.models.backbones.MobileNetV3_small_x1_25(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 1.25，且 model_name为 'small'。

```python
paddleseg.models.backbones.MobileNetV3_large_x0_35(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.35，且 model_name为 'large'。

```python
paddleseg.models.backbones.MobileNetV3_large_x0_5(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.5，且 model_name为 'large'。

```python
paddleseg.models.backbones.MobileNetV3_large_x0_75(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 0.75，且 model_name为 'large'。

```python
paddleseg.models.backbones.MobileNetV3_large_x1_0(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 1.0，且 model_name为 'large'。

```python
paddleseg.models.backbones.MobileNetV3_large_x1_25(**args)
```
> 返回 MobileNetV3 类的一个对象，其scale为 1.25，且 model_name为 'large'。



## [XceptionDeeplab](../../../paddleseg/models/backbones/xception_deeplab.py)
XceptionDeeplab 即 Xception backbone of DeepLabV3+，源于 ["Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"](https://arxiv.org/abs/1802.02611)
```python
class paddleseg.models.backbones.XceptionDeeplab(
            backbone, 
            pretrained = None, 
            output_stride = 16
)
```

### 参数
* **backbone** (str): 选择哪种类型的 Xception_DeepLab。该参数应为('xception_41', 'xception_65', 'xception_71')之一。
* **pretrained** (str, optional): 预训练模型的路径。
* **output_stride** (int, optional): 与输入图像相比，输出特征的步长。该参数应为 8 或 16。 *默认: 16*
```python
paddleseg.models.backbones.Xception41_deeplab(**args)
```
> 返回一个 XceptionDeeplab 类的对象，其层数为41。

```python
paddleseg.models.backbones.Xception65_deeplab(**args)
```
> 返回一个 XceptionDeeplab 类的对象，其层数为65。

```python
paddleseg.models.backbones.Xception71_deeplab(**args)
```
> 返回一个 XceptionDeeplab 类的对象，其层数为71。

