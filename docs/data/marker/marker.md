# 数据格式说明

PaddleSeg采用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。

**NOTE:** 标注图像请使用PNG无损压缩格式的图片。标注类别最多为256类。

## 灰度标注vs伪彩色标注
一般的分割库使用单通道灰度图作为标注图片，往往显示出来是全黑的效果。灰度标注图的弊端：
1. 对图像标注后，无法直接观察标注是否正确。
2. 模型测试过程无法直接判断分割的实际效果。

**PaddleSeg支持伪彩色图作为标注图片，在原来的单通道图片基础上，注入调色板。在基本不增加图片大小的基础上，却可以显示出彩色的效果。** 

同时PaddleSeg也兼容灰度图标注，用户原来的灰度数据集可以不做修改，直接使用。

## 灰度标注转换为伪彩色标注
如果用户需要转换成伪彩色标注图，可使用我们的转换工具。适用于以下两种常见的情况：
1. 如果您希望将指定目录下的所有灰度标注图转换为伪彩色标注图，则执行以下命令，指定灰度标注所在的目录即可。
```buildoutcfg
python pdseg/tools/gray2pseudo_color.py <dir_or_file> <output_dir>
```

|参数|用途|
|-|-|
|dir_or_file|指定灰度标注所在目录|
|output_dir|彩色标注图片的输出目录|

2. 如果您仅希望将指定数据集中的部分灰度标注图转换为伪彩色标注图，则执行以下命令，需要已有文件列表，按列表读取指定图片。
```buildoutcfg
python pdseg/tools/gray2pseudo_color.py <dir_or_file> <output_dir> --dataset_dir <dataset directory> --file_separator <file list separator>
```
|参数|用途|
|-|-|
|dir_or_file|指定文件列表路径|
|output_dir|彩色标注图片的输出目录|
|--dataset_dir|数据集所在根目录|
|--file_separator|文件列表分隔符|
