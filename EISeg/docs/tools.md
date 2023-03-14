简体中文 | [English](tools_en.md)

# 工具脚本

以下内容为EISeg中的相关工具使用。位置位于`EISeg/tool`。

## EISeg PaddleX 语义分割数据集构建

在使用EISeg对网络爬取的图像标注完成后，通过`tool`中的`eiseg2paddlex`，可以将EISeg标注好的数据快速转换为PaddleX的训练格式。使用以下方法：
```
python eiseg2paddlex.py -d save_folder_path -o image_folder_path [-l label_folder_path] [-m mode -s split_rate]
```
其中:
- `save_folder_path`: 为需要保存PaddleX数据的路径，必填。
- `image_folder_path`: 为图像的路径，必填。
- `label_folder_path`: 为标签的路径，非必填，若不填则为自动保存的位置（`image_folder_path/label`）。
- `mode`：数据集的类型，非必填，默认为“seg”，若是目标检测数据集则可使用“det”。
- `split_rate`: 训练集和验证集划分的比例，非必填，若不填则为0.9。

![68747470733a2f2f73332e626d702e6f76682f696d67732f323032312f31302f373134633433396139633766613439622e706e67](https://user-images.githubusercontent.com/71769312/141392744-f1a27774-2714-43a2-8808-2fc14a5a6b5a.png)

## 语义标签转实例标签

语义分割标签转实例分割标签（原标签为0/255），结果为单通道图像采用调色板调色。通过`tool`中的`semantic2instance`，可以将EISeg标注好的语义分割数据转为实例分割数据。使用以下方法：

``` shell
python semantic2instance.py -o label_path -d save_path
```

其中:

- `label_path`: 语义标签存放路径，必填。
- `save_path`: 实例标签保存路径，必填。

![68747470733a2f2f73332e626d702e6f76682f696d67732f323032312f30392f303038633562373638623765343737612e706e67](https://user-images.githubusercontent.com/71769312/141392781-d99ec177-f445-4336-9ab2-0ba7ae75d664.png)

## 视频切分脚本

由于视频数据计算量巨大，为了防止显存不足，推荐将视频切分成100帧以内再标注，脚本位置为`EISeg/tool/cut_video.py`。

## 医疗切片图转换成视频脚本

3D医疗标注是基于视频标注算法来实现的，因此在医疗图像标注前，需要将医疗图像转换成`mp4`格式后再进行标注，脚本位置为`EISeg/tool/medical2video.py`。

## json格式标签转换成coco格式标签

EISeg在标签保存后的```label```文件夹下会生成一个```labelme```的文件夹，里面是与labelme具有相同格式的json文件和```labels.txt```文件，此时如果想把json文件转换成和labelme相同格式的coco文件，可以执行如下命令：

``` shell
python labelme-json2labelme-coco.py label_path save_path --labels txt_path.

# 例如
# python labelme-json2labelme-coco.py mydata/label/labelme/ mydata/label/labelme/output --labels mydata/label/labelme/labels.txt
```

其中:

- `label_path`: labelme格式的json标签保存的路径，必填。
- `save_path`: 转换后的coco格式标签保存的路径，必填。
- `txt_path`: labels.txt文件路径，必填。

转换完成后会在```save_path```中生成三个文件，分别是```annotations.json```, ```JPEGImages```, ```Visualization```，格式同labelme。
