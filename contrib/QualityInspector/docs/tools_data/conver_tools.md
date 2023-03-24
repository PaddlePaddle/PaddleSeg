# convert_tools使用说明

## 1 工具说明

convert_tools是PaddleIndustry提供的用于处理标注文件格式转换的工具集，位于`tools/convert_tools/`目录。由于工业质检可能采用检测，分割或者检测结合RoI分割的解决方案, 为了便于进行PPL的对比和尝试，进行一些简单的任务输入格式的文件处理工作。

*请注意，convert_tools目前为一项实验性功能，若您在使用过程中遇到问题，请及时向我们反馈。*

## 2 文件说明

目前convert_tools共有3个文件，各文件及其功能如下：

- `convert_mask_to_RoI.py`:       全图分割转换为RoI区域分割；
- `convert_mask_to_coco.py`:      转换分割数据为coco格式的json文件，便于训练检测/实例分割；
- `convert_coco_to_RoI_mask.py`:  将coco格式的json文件转化为RoI区域的前景分割。


## 3 使用示例

### 3.1 全图分割标签转RoI分割

使用`convert_mask_to_RoI.py`可以将全图分割转化为RoI分割，方便检测+RoI分割的质检PPL中，RoI分割模块数据的转换，输出图像，标签和txt文件。

#### 3.1.1 命令演示

执行如下命令，完成转化，RoI图像，标签以及RoI.txt保存在output路径中：

```
python3 tools/convert_tools/convert_mask_to_RoI.py \
        --image_path ./dataset/MT_dataset/images/val \
        --anno_path ./dataset/MT_dataset/anno/val \
        --class_num 5 \
        --output_path output \
        --suffix .png
        --to_binary
```

#### 3.1.2 参数说明


| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--image_path` |  需要转化的原图路径                    |           |
| `--anno_path`  |  原图对应的分割标签路径                 |           |
| `--class_num`  |  分割类别数                           |           |
| `--to_binary`  |  是否转化为2类训练                     | `False`   |
| `--suffix`     |  分割标签相对于图像的后缀               |  `" "`    |
| `--pad_scale`  |  剪切图像pad尺度                      |  `0.5`    |
| `--output_path`|  保存路径                             |`./output/`|



#### 3.1.3 结果说明

在`output/images/` 路径下保存RoI图像，注：输出的图像名为原始图像名结合类别id和RoI id;
在`output/annos/` 路径下保存RoI对应的分割标签，0是背景，1是前景；
`output/RoI.txt` 图像和标签列表，每一行表示对应的RoI图像和标签图像路径。

### 3.2 分割标签转coco格式json文件

使用`convert_mask_to_coco.py`可以让图像分割任务数据格式转化为coco json文件，方便用于检测任务。

#### 3.2.1 命令演示

执行如下命令，完成图像分割到coco格式json文件的转换：

```
python3 tools/convert_tools/convert_mask_to_coco.py \
        --image_path dataset/MT_dataset/images/train \
        --anno_path dataset/MT_dataset/annos/train \
        --class_num 5 \
        --label_file dataset/MT_dataset/mt_catIDs.json \
        --output_name dataset/MT_dataset/train.json \
        --suffix .png
```

#### 3.2.2 参数说明

| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--image_path` |  需要转化的原图路径                    |           |
| `--anno_path`  |  原图对应的分割标签路径                 |           |
| `--class_num`  |  分割类别数                           |           |
| `--label_file` |  分割类别id与类别名字典的json文件        |           |
| `--suffix`     |  分割标签相对于图像的后缀               |  `" "`    |
| `--output_name`|  保存json的文件名                     |  `coco`   |

其中，`anno_path`中保存与图像一一对应的分割标签，0为背景， `class_num`是除背景外的类别数，`label_file`是一个json文件，保存了类别id与类别名称的字典，如果不设置label_file,则生成的coco格式的json文件中，categories字段里面类别的`name`是类别序号`str(class_id)`，`suffix`表示标签图像的名字相对于图像名的后缀，包含扩展名，本例中图像名为`xxx.jpg`, 标签名为`xxx.png`,则suffix为`.png`。

#### 3.2.3 结果说明

生成`coco.json`， 包含`images`，`annotations`,`categories`字段。

### 3.3 coco格式json文件标签转RoI前景二值分割

使用`convert_coco_to_RoI_mask.py`可以将检测任务的coco格式json文件转化为RoI分割标签，方便检测+RoI分割的质检PPL中，RoI分割模块数据的转换，输出RoI图像，标签和txt文件。

#### 3.3.1 命令演示

执行如下命令，完成转化，RoI图像，标签以及RoI.txt保存在output路径中：

```
python3 tools/convert_tools/convert_coco_to_RoI_mask.py \
        --json_path dataset/MT_dataset/train.json \
        --seg_classid 1 2 \
        --output_path output/RoI/train/ \
        --suffix .png
```

#### 3.3.2 参数说明


| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--json_path`  |  需要转化的json文件路径                |           |
| `--image_path` |  图像的目录                           |           |
| `--seg_classid`|  需要进行分割的类别                    |           |
| `--suffix`     |  分割标签相对于图像的后缀               |  `" "`    |
| `--pad_scale`  |  剪切图像pad尺度                      |  `0.5`    |
| `--output_path`|  保存路径                             |`./output/`|

其中，`image_path`是图像的相对json文件中路径的根路径， `seg_classid`是需要进行二值RoI分割的类别编号list，`suffix`表示标签图像的名字相对于图像名的后缀，包含扩展名，本例中图像名为`xxx.jpg`, 标签名为`xxx.png`,则suffix为`.png`。

#### 3.3.3 结果说明

在`output/RoI/train/images/` 路径下保存RoI图像，注：输出的图像名为原始图像名结合RoI id;
在`output/RoI/train/annos/` 路径下保存RoI对应的分割标签，0是背景，1是前景；
`output/RoI/train/RoI.txt` 图像和标签列表，每一行表示对应的RoI图像和标签图像路径。
