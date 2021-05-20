# 配置/数据校验
对用户自定义的数据集和yaml配置进行校验，帮助用户排查基本的数据和配置问题。

数据校验脚本如下，支持通过`YAML_FILE_PATH`来指定配置文件。
```
# YAML_FILE_PATH为yaml配置文件路径
python pdseg/check.py --cfg ${YAML_FILE_PATH}
```
运行后，命令行将显示校验结果的概览信息，详细的错误信息可到detail.log文件中查看。

### 1 列表分割符校验
判断在`TRAIN_FILE_LIST`，`VAL_FILE_LIST`和`TEST_FILE_LIST`列表文件中的分隔符`DATASET.SEPARATOR`设置是否正确。
### 2 数据集读取校验
通过是否能成功读取`DATASET.TRAIN_FILE_LIST`，`DATASET.VAL_FILE_LIST`，`DATASET.TEST_FILE_LIST`中所有图片，判断这3项设置是否正确。

若不正确返回错误信息。错误可能有多种情况，如数据集路径设置错误、图片损坏等。

### 3 标注格式校验
检查标注图像是否为PNG格式。

**NOTE:** 标注图像请使用PNG无损压缩格式的图片，若使用其他格式则可能影响精度。

### 4 标注通道数校验
检查标注图的通道数。正确的标注图应该为单通道图像。

### 5 标注类别校验
检查实际标注类别是否和配置参数`DATASET.NUM_CLASSES`，`DATASET.IGNORE_INDEX`匹配。

**NOTE:**
标注图像类别数值必须在[0~(`DATASET.NUM_CLASSES`-1)]范围内或者为`DATASET.IGNORE_INDEX`。
标注类别最好从0开始，否则可能影响精度。

### 6 标注像素统计
统计每种类别的像素总数和所占比例，显示以供参考。统计结果如下：
```
Doing label pixel statistics:
(label class, total pixel number, percentage) = [(0, 2048984, 0.5211), (1, 1682943, 0.428), (2, 197976, 0.0503), (3, 2257, 0.0006)]
```
### 7 图像格式校验
检查图片类型`DATASET.IMAGE_TYPE`是否设置正确。

**NOTE:** 当数据集包含三通道图片时`DATASET.IMAGE_TYPE`设置为rgb；
当数据集全部为四通道图片时`DATASET.IMAGE_TYPE`设置为rgba；

### 8 图像最大尺寸统计
统计数据集中图片的最大高和最大宽，显示以供参考。

### 9 图像与标注图尺寸一致性校验
验证图像尺寸和对应标注图尺寸是否一致。

### 10 模型验证参数`EVAL_CROP_SIZE`校验
验证`EVAL_CROP_SIZE`是否设置正确，共有3种情形：

- 当`AUG.AUG_METHOD`为unpadding时，`EVAL_CROP_SIZE`的宽高应不小于`AUG.FIX_RESIZE_SIZE`的宽高。

- 当`AUG.AUG_METHOD`为stepscaling时，`EVAL_CROP_SIZE`的宽高应不小于原图中最大的宽高。

- 当`AUG.AUG_METHOD`为rangescaling时，`EVAL_CROP_SIZE`的宽高应不小于缩放后图像中最大的宽高。

### 11 数据增强参数`AUG.INF_RESIZE_VALUE`校验
验证`AUG.INF_RESIZE_VALUE`是否在[`AUG.MIN_RESIZE_VALUE`~`AUG.MAX_RESIZE_VALUE`]范围内。若在范围内，则通过校验。
