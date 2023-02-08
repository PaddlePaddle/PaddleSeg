# 一、图像分类（快速在 PaddleLabel 标注的花朵分类数据集上展示如何应用 PaddleX 训练 MobileNetV3_ssld 网络）

PaddleLabel 标注数据+PaddleX 训练预测=快速完成一次图像分类的任务

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的花朵数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`dataset/flower_clas_dataset`，其目录结构如下：

```
├── flower_clas_dataset
│   ├── image
│   │   ├── flower1.jpg
│   │   ├── flower2.jpg
│   │   ├── ...
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── val_list.txt
```

## 2. 训练

### 2.1 安装必备的库

**2.1.1 安装 paddlepaddle**

```
# 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 您的机器是CPU，请运行以下命令安装
# pip install paddlepaddle
```

**2.1.2 安装 PaddleX 以及依赖项**

```
pip install "paddlex<=2.0.0"
pip install scikit-image
pip install threadpoolctl==2.0.0 -i https://mirror.baidu.com/pypi/simple
pip install scikit-learn==0.23.2
```

### 2.2 准备训练

**2.2.1 配置 GPU**

```
# jupyter中使用paddlex需要设置matplotlib
import matplotlib
matplotlib.use('Agg')
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

**2.2.2 定义图像处理流程 transforms**

定义训练和验证过程中，图像的处理流程，其中训练过程包括了部分数据增强操作（验证时不需要），如在本示例中，训练过程使用了`RandomCrop`和`RandomHorizontalFlip`两种数据增强方式，更多图像预处理流程 transforms 的使用可参见[paddlex.cls.transforms](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/cls_transforms.html)

```
from paddlex import transforms as T
train_transforms = T.Compose([
    T.RandomCrop(crop_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])
```

**2.2.3 定义数据集 Dataset**
使用 PaddleX 内置的数据集读取器读取训练和验证数据集。在图像分类中使用`ImageNet`格式的数据集，因此这里采用`pdx.datasets.ImageNet`来加载数据集，该接口的介绍可参见文档[paddlex.datasets.ImageNet](https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-imagenet)

```
train_dataset = pdx.datasets.ImageNet(
    data_dir='./dataset/flower_clas_dataset',
    file_list='./dataset/flower_clas_dataset/train_list.txt',
    label_list='./dataset/flower_clas_dataset/labels.txt',
    transforms=train_transforms)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='./dataset/flower_clas_dataset',
    file_list='./dataset/flower_clas_dataset/val_list.txt',
    label_list='./dataset/flower_clas_dataset/labels.txt',
    transforms=eval_transforms)
```

### 2.3 模型开始训练

在定义好数据集后，即可选择分类模型（这里使用了`MobileNetV3_large_ssld`模型），开始进行训练。
更多模型训练参数介绍可参见文档[paddlex.cls.MobileNetV3_large_ssld](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/classification.html#train)，在如下代码中，模型训练过程每间隔`save_interval_epochs`轮会保存一次模型在`save_dir`目录下，同时在保存的过程中也会在验证数据集上计算相关指标，模型训练过程中相关日志的含义可参见[文档](https://paddlex.readthedocs.io/zh_CN/develop/appendix/metrics.html#id3)

```
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_large_ssld(num_classes=num_classes)
model.train(num_epochs=12,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[6, 8],
            save_interval_epochs=3,
            learning_rate=0.00625,
            save_dir='output/mobilenetv3_large_ssld'
            )
```

## 3. 预测

### 3.1 预测

```
import paddlex as pdx
model = pdx.load_model('./output/mobilenetv3_large_ssld/best_model')
image_path = './dataset/flower_clas_dataset/image/1008566138_6927679c8a.jpg'
result = model.predict(image_path)
print("Predict Result:", result)
```

预测的样例图片是：

<img src="https://ai-studio-static-online.cdn.bcebos.com/c737099ed27f48adac3e33497ecc4cfcddad0df2169c479d9ad98aadfdb9c400" width="50%" height="50%">

预测的结果是：

> Predict Result: \[{'category_id': 0, 'category': 'sunflower', 'score': 0.9999815}\]
> 最终结论：预测正确 ✔

______________________________________________________________________

# 二、目标检测（快速在 PaddleLabel 标注的道路标志检测数据集上展示如何应用 PaddleX 训练 YOLOv3 网络

PaddleLabel 标注数据+PaddleX 训练预测=快速完成一次目标检测的任务

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的路标数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`dataset/roadsign_det_dataset`，其目录结构如下：

```
├── roadsign_det_dataset
│   ├── Annotations
│   ├── JPEGImages
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── val_list.txt
```

## 2. 训练

### 2.1 安装必备的库

**2.1.1 安装 paddlepaddle**

```
# 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 您的机器是CPU，请运行以下命令安装
# pip install paddlepaddle
```

**2.1.2 安装 PaddleX**

```
pip install "paddlex<=2.0.0" -i https://mirror.baidu.com/pypi/simple
```

### 2.2 准备训练

**2.2.1 配置 GPU**

```
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

**2.2.2 定义图像处理流程 transforms**

定义数据处理流程，其中训练和测试需分别定义，训练过程包括了部分测试过程中不需要的数据增强操作，如在本示例中，训练过程使用了`MixupImage`、`RandomDistort`、`RandomExpand`、`RandomCrop`和`RandomHorizontalFlip`共 5 种数据增强方式，更多图像预处理流程[paddlex.det.transforms](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html)

```
from paddlex import transforms as T
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=250),
    T.RandomDistort(),
    T.RandomExpand(),
    T.RandomCrop(),
    T.Resize(target_size=608, interp='RANDOM'),
    T.RandomHorizontalFlip(),
    T.Normalize()])

eval_transforms = T.Compose([
    T.Resize(target_size=608, interp='CUBIC'),
    T.Normalize()
])
```

**2.2.3 定义数据集 Dataset**

目标检测可使用`VOCDetection`格式和`COCODetection`两种数据集，此处由于数据集为 VOC 格式，因此采用 p`dx.datasets.VOCDetection`来加载数据集，该接口的介绍可参见文档[paddlex.datasets.VOCDetection](https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection)

```
train_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset/roadsign_det_dataset',
    file_list='./dataset/roadsign_det_dataset/train_list.txt',
    label_list='./dataset/roadsign_det_dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='./dataset/roadsign_det_dataset',
    file_list='./dataset/roadsign_det_dataset/val_list.txt',
    label_list='./dataset/roadsign_det_dataset/labels.txt',
    transforms=eval_transforms)
```

### 2.3 模型开始训练

在定义好数据集后，即可选择检测模型（这里使用了`yolov3_darknet53`模型），开始进行训练。
关于检测模型训练，更多参数介绍可参见文档[paddlex.det.YOLOv3](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3)，在如下代码中，模型训练过程每间隔`save_interval_epochs`轮会保存一次模型在`save_dir`目录下，同时在保存的过程中也会在验证数据集上计算相关指标，模型训练过程中相关日志的含义可参见[文档](https://paddlex.readthedocs.io/zh_CN/develop/appendix/metrics.html#yolov3)

```
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_interval_epochs=20,
    save_dir='output/yolov3_darknet53')
```

## 3. 预测

### 3.1 预测

使用模型进行预测，同时使用`pdx.det.visualize`将结果可视化，可视化结果将保存到`./output/yolov3_mobilenetv1`下，其中`threshold`代表`Box`的置信度阈值，将`Box`置信度低于该阈值的框过滤不进行可视化

```
import paddlex as pdx
model = pdx.load_model('output/yolov3_darknet53/best_model')
image_path = './dataset/roadsign_det_dataset/JPEGImages/road554.png'
result = model.predict(image_path)
pdx.det.visualize(image_path, result, threshold=0.5, save_dir='./output/yolov3_darknet53')
```

预测的样例图片如下图：

<img src="https://ai-studio-static-online.cdn.bcebos.com/8fb35c64f3424a098858a3f75255f0d56c6f9c9d7e24438c8d1bc2cd71e838d4" width="50%" height="50%">

预测的结果是：

> speedlimit 0.77 预测正确 ✔

______________________________________________________________________

# 三、图像分割（快速在 PaddleLabel 标注的狗子分割数据集上展示如何应用 PaddleX 训练 DeepLabV3 网络）

PaddleLabel 标注数据+PaddleX 训练预测=快速完成一次图像语义分割的任务

## 1. 数据准备

- 首先使用`PaddleLabel`对自制的狗子数据集进行标注，其次使用`Split Dataset`功能分割数据集，最后导出数据集
- 从`PaddleLabel`导出后的内容全部放到自己的建立的文件夹下，例如`dataset/dog_seg_dataset`，其目录结构如下：

```
├── dog_seg_dataset
│   ├── Annotations
│   ├── JPEGImages
│   ├── labels.txt
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── val_list.txt
```

## 2. 训练

### 2.1 安装必备的库

**2.1.1 安装 paddlepaddle**

```
# 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 您的机器是CPU，请运行以下命令安装
# pip install paddlepaddle
```

**2.1.2 安装 PaddleX**

```
pip install "paddlex<=2.0.0" -i https://mirror.baidu.com/pypi/simple
```

### 2.2 准备训练

**2.2.1 配置 GPU**

```
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

**2.2.2 定义图像处理流程 transforms**

定义数据处理流程，其中训练和测试需分别定义，训练过程包括了部分测试过程中不需要的数据增强操作，如在本示例中，训练过程使用了`RandomHorizontalFlip`这种数据增强方式，更多图像预处理流程 transforms 的使用可参见[paddlex.seg.transforms](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/seg_transforms.html)

```
from paddlex import transforms as T
train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.Resize(target_size=512),
    T.Normalize()
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize()
])
```

**2.2.3 定义数据集 Dataset**
语义分割使用`SegDataset`格式的数据集，因此采用`pdx.datasets.SegDataset`来加载数据集，该接口的介绍可参见文档[paddlex.datasets.SegDataset](https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-segdataset)

```
train_dataset = pdx.datasets.SegDataset(
    data_dir='./dataset/dog_seg_dataset',
    file_list='./dataset/dog_seg_dataset/train_list.txt',
    label_list='./dataset/dog_seg_dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='./dataset/dog_seg_dataset',
    file_list='./dataset/dog_seg_dataset/val_list.txt',
    label_list='./dataset/dog_seg_dataset/labels.txt',
    transforms=eval_transforms)
```

### 2.3 模型开始训练

在定义好数据集后，即可选择分割模型（这里使用了`deeplabv3`模型），开始进行训练。

更多训练模型的参数介绍可参见文档[paddlex.seg.DeepLabv3](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)，在如下代码中，模型训练过程每间隔`save_interval_epochs`轮会保存一次模型在`save_dir`目录下，同时在保存的过程中也会在验证数据集上计算相关指标，模型训练过程中相关日志的含义可参见[文档](https://paddlex.readthedocs.io/zh_CN/develop/appendix/metrics.html#id9)

```
num_classes = len(train_dataset.labels)
model = pdx.seg.DeepLabV3P(num_classes=num_classes, backbone='ResNet50_vd')
model.train(
    num_epochs=40,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_interval_epochs=1,
    save_dir='output/deeplab',
    use_vdl=True)
```

## 3. 预测

### 3.1 预测

使用模型进行预测，同时使用`pdx.seg.visualize`将结果可视化，可视化结果将保存到`./output/deeplab`下，其中`weight`代表原图的权重，即 mask 可视化结果与原图权重因子。

```
import paddlex as pdx
model = pdx.load_model('output/deeplab/best_model')
image_name = './dataset/dog_seg_dataset/JPEGImages/e619b17a9c1b9f085dc2712eb603171f.jpeg'
result = model.predict(image_name)
pdx.seg.visualize(image_name, result, weight=0.4, save_dir='./output/deeplab')
```

可视化结果如下所示：

<img src="https://ai-studio-static-online.cdn.bcebos.com/fec970f0e0fd4ddd96ad3d07b318d24c4f004376597946efbed4a599b652ffda" width="50%" height="50%">
<img src="https://ai-studio-static-online.cdn.bcebos.com/783581c1e2f345029cccfc382e0dedc70b58f9b48120467383c923a7ab0401a7" width="50%" height="50%">

## AI Studio 第三方教程推荐

[快速体验演示案例](https://aistudio.baidu.com/aistudio/projectdetail/4383953)
