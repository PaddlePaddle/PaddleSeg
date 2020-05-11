# transforms.transforms

对用于分割任务的数据进行操作。可以利用[Compose](#compose)类将图像预处理/增强操作进行组合。


## Compose类
```python
transforms.transforms.Compose(transforms)
```
根据数据预处理/数据增强列表对输入数据进行操作。
### 参数
* **transforms** (list): 数据预处理/数据增强列表。


## RandomHorizontalFlip类
```python
transforms.transforms.RandomHorizontalFlip(prob=0.5)
```
以一定的概率对图像进行水平翻转,模型训练时的数据增强操作。
### 参数
* **prob** (float): 随机水平翻转的概率。默认值为0.5。


## RandomVerticalFlip类
```python
transforms.transforms.RandomVerticalFlip(prob=0.1)
```
以一定的概率对图像进行垂直翻转,模型训练时的数据增强操作。
### 参数
* **prob**  (float): 随机垂直翻转的概率。默认值为0.1。


## Resize类
```python
transforms.transforms.Resize(target_size, interp='LINEAR')
```
调整图像大小（resize）。

- 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
- 当目标大小（target_size）类型为list或tuple时，根据插值方式，
  将图像resize为target_size, target_size的输入应为[w, h]或（w, h）。
### 参数
* **target_size** (int|list|tuple): 目标大小
* **interp** (str): resize的插值方式，与opencv的插值方式对应，
可选的值为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为"LINEAR"。


## ResizeByLong类
```python
transforms.transforms.ResizeByLong(long_size)
```
对图像长边resize到固定值，短边按比例进行缩放。
### 参数
* **long_size** (int): resize后图像的长边大小。


## ResizeRangeScaling类
```python
transforms.transforms.ResizeRangeScaling(min_value=400, max_value=600)
```
对图像长边随机resize到指定范围内，短边按比例进行缩放,模型训练时的数据增强操作。
### 参数
* **min_value** (int): 图像长边resize后的最小值。默认值400。
* **max_value** (int): 图像长边resize后的最大值。默认值600。


## ResizeStepScaling类
```python
transforms.transforms.ResizeStepScaling(min_scale_factor=0.75, max_scale_factor=1.25, scale_step_size=0.25)
```
对图像按照某一个比例resize，这个比例以scale_step_size为步长，在[min_scale_factor, max_scale_factor]随机变动,模型训练时的数据增强操作。
### 参数
* **min_scale_factor**（float), resize最小尺度。默认值0.75。
* **max_scale_factor** (float), resize最大尺度。默认值1.25。
* **scale_step_size** (float), resize尺度范围间隔。默认值0.25。


## Clip类
```python
transforms.transforms.Clip(min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0])
```
对图像上超出一定范围的数据进行裁剪。

### 参数
* **min_var** (list): 裁剪的下限，小于min_val的数值均设为min_val. 默认值[0, 0, 0].
* **max_var** (list): 裁剪的上限，大于max_val的数值均设为max_val. 默认值[255.0, 255.0, 255.0]


## Normalize类
```python
transforms.transforms.Normalize(min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```
对图像进行标准化。

1.图像像素归一化到区间 [0.0, 1.0]。
2.对图像进行减均值除以标准差操作。
### 参数
* **min_val** (list): 图像数据集的最小值。默认值[0, 0, 0].
* **max_val** (list): 图像数据集的最大值。默认值[255.0, 255.0, 255.0]
* **mean** (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5]。
* **std** (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5]。


## Padding类
```python
transforms.transforms.Padding(target_size, im_padding_value=127.5, label_padding_value=255)
```
对图像或标注图像进行padding，padding方向为右和下。根据提供的值对图像或标注图像进行padding操作。
### 参数
* **target_size** (int|list|tuple): padding后图像的大小。
* **im_padding_value** (list): 图像padding的值。默认为127.5
* **label_padding_value** (int): 标注图像padding的值。默认值为255（仅在训练时需要设定该参数）。


## RandomPaddingCrop类
```python
transforms.transforms.RandomPaddingCrop(crop_size=512, im_padding_value=127.5, label_padding_value=255)
```
对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作，模型训练时的数据增强操作。
### 参数
* **crop_size**（int|list|tuple): 裁剪图像大小。默认为512。
* **im_padding_value** (list): 图像padding的值。默认为127.5。
* **label_padding_value** (int): 标注图像padding的值。默认值为255。


## RandomBlur类
```python
transforms.transforms.RandomBlur(prob=0.1)
```
以一定的概率对图像进行高斯模糊，模型训练时的数据增强操作。
### 参数
* **prob** (float): 图像模糊概率。默认为0.1。


## RandomScaleAspect类
```python
transforms.transforms.RandomScaleAspect(min_scale=0.5, aspect_ratio=0.33)
```
裁剪并resize回原始尺寸的图像和标注图像,模型训练时的数据增强操作。

按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。
### 参数
* **min_scale**  (float)：裁取图像占原始图像的面积比，取值[0，1]，为0时则返回原图。默认为0.5。
* **aspect_ratio** (float): 裁取图像的宽高比范围，非负值，为0时返回原图。默认为0.33。
