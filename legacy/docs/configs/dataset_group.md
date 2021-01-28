# cfg.DATASET

DATASET Group存放所有与数据集相关的配置

## `DATA_DIR`

数据集主目录，PaddleSeg在读取数据文件列表时，会将列表中的文件名与主目录拼接得到图片的绝对路径

### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `TRAIN_FILE_LIST`

训练集列表，调用`pdseg/train.py`进行训练时，会读取该列表中的图片进行训练

文件列表由多行组成，每一行的格式为
```
<img_path><sep><label_path>
```
### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `VAL_FILE_LIST`

验证集列表，调用`pdseg/eval.py`进行效果评估时，会读取该列表中的图片进行评估

文件列表由多行组成，每一行的格式为
```
<img_path><sep><label_path>
```

### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `TEST_FILE_LIST`

测试集列表，调用`pdseg/vis.py`进行可视化展示时，会读取该列表中的图片进行预测

文件列表由多行组成，每一行的格式为
```
<img_path><sep><label_path>
```

### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `VIS_FILE_LIST`

可视化列表，调用`pdseg/train.py`进行训练时，如果打开了--use_vdl开关，则在每次模型保存的时候，会读取该列表中的图片进行可视化

文件列表由多行组成，每一行的格式为
```
<img_path><sep><label_path>
```

### 默认值

无（需要用户自己填写）

<br/>
<br/>

## `NUM_CLASSES`

类别数量，构建网络所需

### 默认值

19（但是一般需要用户修改为自己数据集的类别数量）

### 注意事项

数据集中的label标注必须为0 ~ NUM_CLASSES - 1，如果label设置错误，会导致计算IOU时出现异常

<br/>
<br/>

## `IMAGE_TYPE`

图片类型，支持`rgb`、`rgba`、`gray`三种格式

### 默认值

`rgb`

<br/>
<br/>

## `SEPARATOR`

文件列表中用于分隔输入图片和标签图片的分隔符

### 默认值

空格符` `

### 例子
假设训练文件列表如下，则 `SEPARATOR` 应该填写 `|`
```
mydata/train/image1.jpg|mydata/train/image1.label.jpg
mydata/train/image2.jpg|mydata/train/image2.label.jpg
mydata/train/image3.jpg|mydata/train/image3.label.jpg
mydata/train/image4.jpg|mydata/train/image4.label.jpg
...
```

<br/>
<br/>

## `IGNORE_INDEX`
需要忽略的像素标签值，label中所有标记为该值的像素不会参与到loss的计算以及IOU、Acc等指标的计算

### 默认值

255
