## 无标注数据集

如果您的数据集不包含任何标注，只需将所有图像文件放在一个文件夹下。 PaddleLabel 会遍历文件夹（及所有子文件夹）中所有文件，并按照**文件拓展名**判断其类型，导入所有图像文件。所有隐藏文件（文件名以`.`开头）将被忽略。

## 基础功能

不同类型项目的数据集文件结构有所不同，但大多数类型的项目都支持一些基础功能。

### labels.txt

所有不使用 COCO 格式保存标注的项目都支持`labels.txt`。PaddleLabel 在导入过程中会在数据集路径下寻找`labels.txt`文件。您可以在这个文件中列出该项目的所有标签（每行一个）。例如下面这样:

```text
# labels.txt
Monkey
Mouse
```

PaddleLabel 的标签名称支持任何字符串，但是标签名称可能被用作导出数据集的文件夹名，所以应避免任何您的操作系统不支持的字符串，可以参考[这篇回答](https://stackoverflow.com/a/31976060)。Paddle 生态中的其他工具对标签名可能有进一步限制，如[PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/data/format/classification.md)不支持中文字符作为标签名称。

在导入过程中，`labels.txt`可以包含标签名以外的信息。目前支持 4 种格式，如下所示。其中`|`表示分隔符，默认为空格。

标签长度：

- 1：标签名
- 2：标签名 | 标签编号
- 3：标签名 | 标签编号 | 十六进制颜色或常用颜色名称或灰度值
- 5：标签名 | 标签编号 | 红色 | 绿色 | 蓝色

其他：

- `//`：`//`后的字符串将被作为标签注释
- `-`：如果需要指定标签颜色，但不想指定标签编号，在标签编号位置写`-`

一些例子：

```text
dog
monkey 4
mouse - #0000ff // mouse's id will be 5
cat 10 yellow
zibra 11 blue // some common colors are supported
snake 12 255 0 0 // rgb color
```

所有支持的颜色名称在[这里](https://github.com/PaddleCV-SIG/PaddleLabel/blob/develop/paddlelabel/task/util/color.py#L15)列出。

在导入过程中，PaddleLabel 会首先创建`labels.txt`中指定的标签。因此这个文件中的标签的编号将从**0**开始并递增。在导出过程中也将生成此文件。

### xx_list.txt

所有不使用 COCO 格式保存标注的项目都支持`xx_list.txt`。`xx_list.txt`包括`train_list.txt`，`val_list.txt`和`test_list.txt`。这三个文件需要放在数据集文件夹的根目录中，与`labels.txt`相同。

这三个文件指定了数据集的划分以及标签或标注文件与图像文件间的匹配关系（比如 voc 格式下，每一行是图像文件的路径和标签文件的路径）。这三个文件的内容结构相同，每一行都以一条数据的路径开始，其路径为相对数据集根目录的相对路径，后面跟着表示类别的整数/字符串，或者标签文件的路径。例如：

```text
# train_list.txt
image/9911.jpg 0 3
image/9932.jpg 4
image/9928.jpg Cat
```

或

```text
# train_list.txt
JPEGImages/1.jpeg Annotations/1.xml
JPEGImages/2.jpeg Annotations/2.xml
JPEGImages/3.jpeg Annotations/3.xml
```

需要注意的是，**大多数项目都只会用到`xx_list.txt`中的数据集划分信息**。

如果标签类别为整数，PaddleLabel 将在`labels.txt`中查找标签，标签 id 从**0**开始。一些数据集中一条数据可以有多个类别，比如图像多分类。如果希望用数字作为标签名称，您可以将数字写在`labels.txt`中，并在`xx_list.txt`中提供标签 id。或者可以给数字标签加一个前缀，例如将`10`表示为`n10`。

这三个文件都将在导出过程中生成，即使其中一些文件是空的。注意，为了确保这些文件可以被 Paddle 生态系统中的其他工具读取，没有注释的数据**不会包含在`xx_list.txt`中**。
