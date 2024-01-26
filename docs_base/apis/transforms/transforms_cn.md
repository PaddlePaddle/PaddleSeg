简体中文 | [English](transforms.md)
# [paddleseg.transforms](../../../paddleseg/transforms/transforms.py)

## [Compose](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Compose(transforms, to_rgb = True)
```

    使用相关预处理与数据增强操作对输入数据进行变换。

    对于所有操作，输入数据的维度为[height, width, channels]。

### 参数
* **transforms** (list): 一个包含数据预处理或增强方法的列表。
* **to_rgb** (bool, optional): 是否将图像转换到RGB颜色空间。默认: True。

### Raises
* **TypeError**: 当'transforms' 不是一个列表时引起。
* **ValueError**: 当'transforms' 元素个数小于1时引起。

## [RandomHorizontalFlip](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomHorizontalFlip(prob = 0.5)
```
    以一定的概率对图像进行水平翻转。

### 参数
* **prob** (float, optional): 水平翻转的概率。 默认: 0.5。

## [RandomVerticalFlip](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomVerticalFlip(prob = 0.1)
```
    以一定的概率对图像进行垂直翻转。

### 参数
* **prob** (float, optional): 垂直翻转的概率。默认: 0.1。

## [Resize](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Resize(target_size = (512, 512), interp='LINEAR')
```
    调整图像大小。

### 参数
* **target_size** (list|tuple, optional): 目标图像的大小。 默认: (512, 512)。
* **interp** (str, optional): resize的插值方式，与opencv的插值方式对应。
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。请注意，当指定为 'RANDOM' 时，将在所有插值模式中进行随机选择。默认: "LINEAR"。

### Raises
* **TypeError**: 当'target_size' 类型既非list，亦非tuple时引起。
* **ValueError**: 当指定的"interp" 不在事先定义的方法集 ('NEAREST', 'LINEAR', 'CUBIC','AREA', 'LANCZOS4','RANDOM')中时引起。

## [ResizeByLong](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeByLong(long_size)
```
    对图像长边resize到固定值，短边按比例进行缩放。

### 参数
* **long_size** (int): 长边的目标值。

## [ResizeRangeScaling](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeRangeScaling(min_value = 400, max_value = 600)
```
    对图像长边随机resize到指定范围内，短边按比例进行缩放。

### 参数
* **min_value** (int, optional): 经resize后长边的最小值。默认: 400。
* **max_value** (int, optional): 经resize后长边的最大值。默认: 600。

## [ResizeStepScaling](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.ResizeStepScaling(min_scale_factor = 0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25)
```

    对图像按照某一个比例进行缩放，这个比例以scale_step_size为步长，在[min_scale_factor, max_scale_factor]随机变动。

### 参数
* **min_scale_factor** (float, optional): 最小scale。 默认: 0.75。
* **max_scale_factor** (float, optional): 最大scale.。默认: 1.25。
* **scale_step_size** (float, optional): scale 步长。默认: 0.25。

### Raises
* **ValueError**: 当 min_scale_factor < max_scale_factor 时引起。

## [Normalize](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
```
    对图像进行标准化。具体操作如下：
    1.像素值减去min_val。
    2.像素值除以(max_val-min_val), 归一化到区间 [0.0, 1.0]。
    3.对图像进行减均值除以标准差操作。

### 参数
* **mean** (list, optional): 图像数据集的均值。默认: [0.5, 0.5, 0.5]。
* **std** (list, optional): 图像数据集的标准差。默认: [0.5, 0.5, 0.5]。
注：在mean与std参数中，列表长度应与图像通道数保持一致。

### Raises
* **ValueError**: 当 mean/std 不是 list ，或std列表中出现了0时引起。

## [Padding](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.Padding(target_size,
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```
    根据提供的值对图像或标注图像进行padding操作。对图像或标注图像进行padding，padding方向为右和下。

### 参数
* **target_size** (list|tuple): padding后的目标尺寸。
* **im_padding_value** (list, optional): 对原始图像的padding值。
            默认: [127.5, 127.5, 127.5]。
* **label_padding_value** (int, optional): 对标注图像的padding值。默认: 255。

### Raises
* **TypeError:** 当 target_size 的类型既非list，亦非tuple时引起。
* **ValueError**: 当 target_size 的长度不是2时引起。  

## [RandomPaddingCrop](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomPaddingCrop(crop_size = (512, 512),
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```
    对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作，padding方向为右和下。

### 参数
* **crop_size** (tuple, optional): 预期裁剪大小。默认: (512, 512)。
* **im_padding_value** (list, optional): 对原始图像的padding值。
            默认: [127.5, 127.5, 127.5]。
* **label_padding_value** (int, optional): 对标注图像的padding值。默认: 255。

### Raises
* **TypeError**: 当 crop_size 的类型既非list，亦非tuple时引起。
* **ValueError**: 当 crop_size 的长度不是2时引起。

## [RandomBlur](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomBlur(prob = 0.1)
```
    以一定的概率对图像进行高斯模糊。

### 参数
* **prob** (float, optional): 图像模糊概率。默认: 0.1.

## [RandomRotation](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomRotation(max_rotation = 15,
                 im_padding_value = (127.5, 127.5, 127.5),
                 label_padding_value = 255)
```
    对图像进行随机旋转，当存在标注图像时，同步进行，并对旋转后的图像和标注图像进行相应的padding。

### 参数
* **max_rotation** (float, optional): 最大旋转角度。 默认: 15。
* **im_padding_value** (list, optional): 对原始图像padding的值。
            默认: [127.5, 127.5, 127.5].
* **label_padding_value** (int, optional): 对标注图像padding的值。默认: 255。
注：参数im_padding_value、label_padding_value的列表长度应与图像通道数保持一致。

## [RandomScaleAspect](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomScaleAspect(min_scale = 0.5, aspect_ratio = 0.33)
```
    裁剪并resize回原始尺寸的图像和标注图像。
    按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。

### 参数
* **min_scale** (float, optional): 裁取图像占原始图像的面积比的最小值，取值范围为[0，1]，等于0时则返回原图。默认: 0.5。
* **aspect_ratio** (float, optional): 裁取图像的宽高比范围的最小值，非负值，为0时返回原图。默认: 0.33。


## [RandomDistort](../../../paddleseg/transforms/transforms.py)
```python
class paddleseg.transforms.RandomDistort(brightness_range = 0.5,
                 brightness_prob = 0.5,
                 contrast_range = 0.5,
                 contrast_prob = 0.5,
                 saturation_range = 0.5,
                 saturation_prob = 0.5,
                 hue_range = 18,
                 hue_prob = 0.5)
```
    对图像按配置参数进行随机的内容变换（扭曲）。

### 参数
* **brightness_range** (float, optional): 明亮度的变化范围。默认: 0.5。
* **brightness_prob** (float, optional): 随机调整明亮度的概率。默认: 0.5。
* **contrast_range** (float, optional): 对比度的变化范围。 默认: 0.5。
* **contrast_prob** (float, optional): 随机调整对比度的概率. 默认: 0.5。
* **saturation_range** (float, optional): 饱和度的变化范围。 默认: 0.5。
* **saturation_prob** (float, optional): 随机调整饱和度的概率。 默认: 0.5。
* **hue_range** (int, optional): 调整色相角度的差值取值范围。 默认: 18。
* **hue_prob** (float, optional): 随机调整色调的概率。 默认: 0.5。
