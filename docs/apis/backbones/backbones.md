English | [简体中文](backbones_cn.md)
# paddleseg.models.backbone

The models subpackage contains backbones extracting features for sementic segmentation models.

- [ResNet_vd](#ResNet_vd)
- [HRNet](#HRNet)
- [MobileNetV3](#MobileNetV3)
- [XceptionDeeplab](#xceptiondeeplab)


## [ResNet_vd](../../../paddleseg/models/backbones/resnet_vd.py)
ResNet_vd backbone from ["Bag of Tricks for Image Classification with Convolutional Neural Networks"](https://arxiv.org/pdf/1812.01187.pdf)
```python
class paddleseg.models.backbones.Resnet_vd(
            layers = 50, 
            output_stride = None, 
            multi_grid = (1, 1, 1), 
            lr_mult_list = (0.1, 0.1, 0.2, 0.2), 
            pretrained = None
)
```

### Args
* **layers** (int, optional): The layers of ResNet_vd. The supported layers are [18, 34, 50, 101, 152, 200]. *Default: 50*
* **output_stride** (int, optional): Compared with the input image, the zoom stride of the output feature, this parameter will affect the downsampling multiple.This parameter should be 8 or 16. It is 8 or 16. *Default: 8*
* **multi_grid** (tuple|list, optional): The grid of stage4. Used to expand the receptive field of convolution. *Defult: (1, 1, 1)*
* **pretrained** (str, optional): The path of pretrained model.
```python
paddleseg.models.backbones.ResNet18_vd(**args)
```
> Return a object of ResNet_vd class which layers is 18.

```python
paddleseg.models.backbones.ResNet34_vd(**args)
```
> Return a object of ResNet_vd class which layers is 34.

```python
paddleseg.models.backbones.ResNet50_vd(**args)
```
> Return a object of ResNet_vd class which layers is 50.

```python
paddleseg.models.backbones.ResNet101_vd(**args)
```
> Return a object of ResNet_vd class which layers is 101.

```python
paddleseg.models.backbones.ResNet152_vd(**args)
```
> Return a object of ResNet_vd class which layers is 152.

```python
padddelseg.models.backbones.ResNet200_vd(**args)
```
> Return a object of ResNet_vd class which layers is 200.

## [HRNet](../../../paddleseg/models/backbones/hrnet.py)
HRNet backbone from ["HRNet：Deep High-Resolution Representation Learning for Visual Recognition"](https://arxiv.org/pdf/1908.07919.pdf)
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

### Args
* **pretrained** (str, optional): The path of pretrained model.
* **stage1_num_modules** (int, optional): Number of modules for stage1. *Default: 1*
* **stage1_num_blocks** (list, optional): Number of blocks per module for stage1.*Default: (4,)*
* **stage1_num_channels** (list, optional): Number of channels per branch for stage1. *Default: (64,)*
* **stage2_num_modules** (int, optional): Number of modules for stage2. *Default 1*
* **stage2_num_blocks** (list, optional): Number of blocks per module for stage2. *Default: (4, 4)*
* **stage2_num_channels** (list, optional): Number of channels per branch for stage2. *Default: (18, 36)*
* **stage3_num_modules** (int, optional): Number of modules for stage3. *Default 4*
* **stage3_num_blocks** (list, optional): Number of blocks per module for stage3. *Default: (4, 4, 4)*
* **stage3_num_channels** (list, optional): Number of channels per branch for stage3. *Default: (18, 36, 72)*
* **stage4_num_modules** (int, optional): Number of modules for stage4. *Default 3*
* **stage4_num_blocks** (list, optional): Number of blocks per module for stage4. *Default: (4, 4, 4, 4)*
* **stage4_num_channels** (list, optional): Number of channels per branch for stage4. *Default: (18, 36, 72, 144)*
* **has_se** (bool, optional): Whether to use Squeeze-and-Excitation module. *Default False*
* **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. *Default: False*
```python
paddleseg.models.backbones.HRNet_W18_Small_V1(**kwargs)
```
> Return a object of HRNet class which width is 18 and it is smaller than HRNet_W18_Small_V2.

```python
paddleseg.models.backbones.HRNet_W18_Small_V2(**kwargs)
```
> Return a object of HRNet class which width is 18 and it is smaller than HRNet_W18.

```python
paddleseg.models.backbones.HRNet_W18(**kwargs)
```
> Return a object of HRNet class which width is 18.

```python
paddleseg.models.backbones.HRNet_W30(**kwargs)
```
> Return a object of HRNet class which width is 30.

```python
paddleseg.models.backbones.HRNet_W32(**kwargs)
```
> Return a object of HRNet class which width is 32.

```python
paddleseg.models.backbones.HRNet_W40(**kwargs)
```
> Return a object of HRNet class which width is 40.

```python
paddleseg.models.backbones.HRNet_W44(**kwargs)
```
> Return a object of HRNet class which width is 44.

```python
paddleseg.models.backbones.HRNet_W48(**kwargs)
```
> Return a object of HRNet class which width is 48.

```python
paddleseg.models.backbones.HRNet_W60(**kwargs)
```
> Return a object of HRNet class which width is 60.

```python
paddleseg.models.backbones.HRNet_W64(**kwargs)
```
> Return a object of HRNet class which width is 64.



## [MobileNetV3](../../../paddleseg/models/backbones/mobilenetv3.py)
MobileNetV3 backbone from ["Searching for MobileNetV3"](https://arxiv.org/pdf/1905.02244.pdf).
```python
class paddleseg.models.backbones.MobileNetV3(
            pretrained = None, 
            scale = 1.0, 
            model_name = "small",
            output_stride = None
)
```

### Args
* **pretrained** (str, optional): The path of pretrained model.
* **scale** (float, optional): The scale of channels. Recommendation: Compared with the small model, set a higher scale for the large model. *Default: 1.0*
* **model_name** (str, optional): Model name. It determines the type of MobileNetV3. The value is 'small' or 'large'. *Defualt: 'small'*
* **output_stride** (int, optional): The stride of output features compared to input images. The value should be one of [2, 4, 8, 16, 32]. *Default: None*
```python
paddleseg.models.backbones.MobileNetV3_small_x0_35(**args)
```
> Return a object of MobileNetV3 class which scale is 0.35 and model_name is small.

```python
paddleseg.models.backbones.MobileNetV3_small_x0_5(**args)
```
> Return a object of MobileNetV3 class which scale is 0.5 and model_name is small.

```python
paddleseg.models.backbones.MobileNetV3_small_x0_75(**args)
```
> Return a object of MobileNetV3 class which scale is 0.75 and model_name is small.

```python
paddleseg.models.backbones.MobileNetV3_small_x1_0(**args)
```
> Return a object of MobileNetV3 class which scale is 1.0 and model_name is small.

```python
paddleseg.models.backbones.MobileNetV3_small_x1_25(**args)
```
> Return a object of MobileNetV3 class which scale is 1.25 and model_name is small.

```python
paddleseg.models.backbones.MobileNetV3_large_x0_35(**args)
```
> Return a object of MobileNetV3 class which scale is 0.35 and model_name is large.

```python
paddleseg.models.backbones.MobileNetV3_large_x0_5(**args)
```
> Return a object of MobileNetV3 class which scale is 0.5 and model_name is large.

```python
paddleseg.models.backbones.MobileNetV3_large_x0_75(**args)
```
> Return a object of MobileNetV3 class which scale is 0.75 and model_name is large.

```python
paddleseg.models.backbones.MobileNetV3_large_x1_0(**args)
```
> Return a object of MobileNetV3 class which scale is 1.0 and model_name is large.

```python
paddleseg.models.backbones.MobileNetV3_large_x1_25(**args)
```
> Return a object of MobileNetV3 class which scale is 1.25 and model_name is large.


## [XceptionDeeplab](../../../paddleseg/models/backbones/xception_deeplab.py)
Xception backbone of DeepLabV3+ from ["Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"](https://arxiv.org/abs/1802.02611)
```python
class paddleseg.models.backbones.XceptionDeeplab(
            backbone, 
            pretrained = None, 
            output_stride = 16
)
```

### Args
* **backbone** (str): Which type of Xception_DeepLab to select. It should be one of ('xception_41', 'xception_65', 'xception_71').
* **pretrained** (str, optional): The path of pretrained model.
* **output_stride** (int, optional): The stride of output features compared to input images. It is 8 or 16. 
*Default: 16*
```python
paddleseg.models.backbones.Xception41_deeplab(**args)
```
> Return a object of XceptionDeeplab class which layers is 41.

```python
paddleseg.models.backbones.Xception65_deeplab(**args)
```
> Return a object of XceptionDeeplab class which layers is 65.

```python
paddleseg.models.backbones.Xception71_deeplab(**args)
```
> Return a object of XceptionDeeplab class which layers is 71.
