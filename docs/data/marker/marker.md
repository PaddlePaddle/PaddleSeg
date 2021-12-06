English|[简体中文](marker_cn.md)
# Preparation of Annotation Data

## 1、Pre-knowledge

### 1.1 Annotation Protocal
PaddleSeg uses a single-channel annotated image, and each pixel value represents a category, and the pixel label category needs to increase from 0. For example, 0, 1, 2, 3 indicate that there are 4 categories.

**NOTE:** Please use PNG lossless compression format for annotated images. The maximum number of label categories is 256.

### 1.2 Grayscale Annotation VS Pseudo-color Annotation
The general segmentation library uses a single-channel grayscale image as the annotated image, and it often shows a completely black effect. Disadvantages of gray scale annotated map:
1. After annotating an image, it is impossible to directly observe whether the annotation is correct.
2. The actual effect of segmentation cannot be directly judged during the model testing process.

**PaddleSeg supports pseudo-color images as annotated images, and injects palettes on the basis of the original single-channel images. On the basis of basically not increasing the size of the picture, it can show a colorful effect.**

At the same time, PaddleSeg is also compatible with gray-scale icon annotations. The user's original gray-scale dataset can be used directly without modification.
![](../image/image-11.png)

### 1.3 Convert grayscale annotations to pseudo-color annotations
If users need to convert to pseudo-color annotation maps, they can use our conversion tool. Applies to the following two common situations:
1. If you want to convert all grayscale annotation images in a specified directory to pseudo-color annotation images, execute the following command to specify the directory where the grayscale annotations are located.
```buildoutcfg
python tools/gray2pseudo_color.py <dir_or_file> <output_dir>
```

|Parameter|Effection|
|-|-|
|dir_or_file|Specify the directory where gray scale labels are located|
|output_dir|Output directory of color-labeled pictures|

2. If you only want to convert part of the gray scale annotated image in the specified dataset to pseudo-color annotated image, execute the following command, you need an existing file list, and read the specified image according to the list.
```buildoutcfg
python tools/gray2pseudo_color.py <dir_or_file> <output_dir> --dataset_dir <dataset directory> --file_separator <file list separator>
```
|Parameter|Effection|
|-|-|
|dir_or_file|Specify the directory where gray scale labels are located|
|output_dir|Output directory of color-labeled pictures|
|--dataset_dir|The root directory where the dataset is located|
|--file_separator|File list separator|


### 1.4 How PaddleSeg uses datasets
We want to write the path of the image to the three folders `train.txt`, `val.txt`, `test.txt` and `labels.txt`, because PaddleSeg locates the image by reading these text files Path.

The texts of `train.txt`, `val.txt` and `test.txt` are divided into two columns with spaces as separators. The first column is the relative path of the image file relative to the dataset, and the second column is the relative path of the image file The relative path of the dataset. As follows:

```
images/xxx1.jpg  annotations/xxx1.png
images/xxx2.jpg  annotations/xxx2.png
...
```
`labels.txt`: Each row has a separate category, and the corresponding row number is the id corresponding to the category (the row number starts from 0), as shown below:
```
labelA
labelB
...
```


## 2、Annotate custom datasets
If you want to use a custom dataset, you need to collect images for training, evaluation, and testing in advance, and then use the data annotation tool to complete the data annotation. If you want to use ready-made datasets such as Cityscapes and Pascal VOC, you can skip this step.

PaddleSeg already supports 2 kinds of labeling tools: `LabelMe`, and `EISeg`. The annotation tutorial is as follows:

- [LabelMe Tutorial](../transform/transform_cn.md)
- [EISeg Tutorial](../../../EISeg/README.md)

After annotating with the above tools, please store all annotated images in the annotations folder, and then proceed to the next step.


## 3、Split a custom dataset

We all know that the training process of neural network models is usually divided into training set, validation set, and test set. If you are using a custom dataset, PaddleSeg supports splitting the dataset by running scripts. If you want to use ready-made datasets such as Cityscapes and Pascal VOC, you can skip this step.

### 3.1 Original image requirements
The size of the original image data should be (h, w, channel), where h, w are the height and width of the image, and channel is the number of channels of the image.

### 3.2 Annotation image requirements
The annotated image must be a single-channel image, the pixel value is the corresponding category, and the pixel annotated category needs to increase from 0.
For example, 0, 1, 2, 3 means that there are 4 categories, and the maximum number of labeled categories is 256. Among them, you can specify a specific pixel value to indicate that the pixel of that value does not participate in training and evaluation (the default is 255).


### 3.3 Custom dataset segmentation and file list generation

For all data that is not divided into training set, validation set, and test set, PaddleSeg provides a script to generate segmented data and generate a file list.
If your dataset has been segmented like Cityscapes, Pascal VOC, etc., please skip to section 4. Otherwise, please refer to the following tutorials:


### Use scripts to randomly split the custom dataset proportionally and generate a file list
The data file structure is as follows:
```
./dataset/  # Dataset root directory
|--images  # Original image catalog
|  |--xxx1.jpg
|  |--...
|  └--...
|
|--annotations  # Annotated image catalog
|  |--xxx1.png
|  |--...
|  └--...
```


Among them, the corresponding file name can be defined according to needs.

The commands used are as follows, which supports enabling specific functions through different Flags.
```
python tools/split_dataset_list.py <dataset_root> <images_dir_name> <labels_dir_name> ${FLAGS}
```
Parameters:
- dataset_root: Dataset root directory
- images_dir_name: Original image catalog
- labels_dir_name: Annotated image catalog

FLAGS:

|FLAG|Meaning|Default|Parameter numbers|
|-|-|-|-|
|--split|Dataset segmentation ratio|0.7 0.3 0|3|
|--separator|File list separator|"&#124;"|1|
|--format|Data format of pictures and label sets|"jpg"  "png"|2|
|--label_class|Label category|'\_\_background\_\_' '\_\_foreground\_\_'|several|
|--postfix|Filter pictures and label sets according to whether the main file name (without extension) contains the specified suffix|""   ""（2 null characters）|2|



After running, `train.txt`, `val.txt`, `test.txt` and `labels.txt` will be generated in the root directory of the dataset.

**Note:** Requirements for generating the file list: either the original image and the number of annotated images are the same, or there is only the original image without annotated images. If the dataset lacks annotated images, a file list without separators and annotated image paths will be generated.

#### Example
```
python tools/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format jpg png
```



## 4、Dataset file organization

PaddleSeg uses a common file list method to organize training set, validation set and test set. The corresponding file list must be prepared before the training, evaluation, and visualization process.

It is recommended to organize it into the following structure:

    custom_dataset
        |
        |--images
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...
        |
        |--labels
        |  |--label1.png
        |  |--label2.png
        |  |--...
        |
        |--train.txt
        |
        |--val.txt
        |
        |--test.txt

### 4.1 File List Specification(Training,Evaluating)

- During training and evaluating, annotated images are required.

- That is, the contents of `train.txt` and `val.txt` are as follows:
    ```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
    ```

Among them, `image1.jpg` and `label1.png` are the original image and its corresponding annotated image, respectively. For the content specification in `test.txt`, please refer to [Section 4.2](#4.2-File-List-Specification-(Predicting)).

**NOTE**

* Make sure that the separator exists only once per line in the file list. If there are spaces in the file name, please use "|" and other unusable characters in the file name to split.

* Please save the file list in **UTF-8** format, PaddleSeg uses UTF-8 encoding to read file_list files by default.

* You need to ensure that the separator of the file list is consistent with your Dataset class. The default separator is a `space`.

### 4.2 File List Specification (Predicting)
- During predicting, the model uses only the original image.

- That is, the content of `test.txt` is as follows:
    ```
    images/image1.jpg
    images/image2.jpg
    ...
    ```

- When calling `predict.py` for visual display, annotated images can be included in the file list. During predicting, the model will automatically ignore the annotated images given in the file list. Therefore, you can make predictions on the training and validatsion datasets without modifying the contents of the `train.txt` and `val.txt` files mentioned in
[Section 4.1](#4.1-File-List-Specification(Training,Evaluating)).


### 4.3 Organize the dataset directory structure

If the user wants to generate a file list of the dataset, it needs to be organized into the following directory structure (similar to the Cityscapes dataset). You can divide it manually, or refer to the method of automatic segmentation using scripts in Section 3.

```
./dataset/   # Dataset root directory
├── annotations      # Annotated image catalog
│   ├── test
│   │   ├── ...
│   │   └── ...
│   ├── train
│   │   ├── ...
│   │   └── ...
│   └── val
│       ├── ...
│       └── ...
└── images       # Original image catalog
    ├── test
    │   ├── ...
    │   └── ...
    ├── train
    │   ├── ...
    │   └── ...
    └── val
        ├── ...
        └── ...
Note:The above directory name can be any
```

### 4.4 Generate file list
PaddleSeg provides a script for generating file lists, which can be applied to custom datasets or cityscapes datasets, and supports different Flags to enable specific functions.
```
python tools/create_dataset_list.py <your/dataset/dir> ${FLAGS}
```
After running, a file list of the training/validation/test set will be generated in the root directory of the dataset (the main name of the file is the same as `--second_folder`, and the extension is `.txt`).

**Note:** Requirements for generating the file list: either the original image and the number of annotated images are the same, or there is only the original image without annotated images. If the dataset lacks annotated images, a file list without separators and annotated image paths can still be automatically generated.

#### FLAGS list

|FLAG|Effection|Default|Parameter numbers|
|-|-|-|-|
|--type|Specify the dataset type, `cityscapes` or `custom`|`custom`|1|
|--separator|File list separator|"&#124;"|1|
|--folder|Folder name for pictures and label sets|"images" "annotations"|2|
|--second_folder|The folder name of the training/validation/test set|"train" "val" "test"|several|
|--format|Data format of pictures and label sets|"jpg"  "png"|2|
|--postfix|Filter pictures and label sets according to whether the main file name (without extension) contains the specified suffix|""   ""（2 null characters）|2|

#### Example
- **For custom datasets**

If you have organized the dataset directory structure according to the above instructions, you can run the following command to generate a file list.

```
# Generate a file list, the separator is a space, and the data format of the picture and the label set is png
python tools/create_dataset_list.py <your/dataset/dir> --separator " " --format jpg png
```
```
# Generate a list of files. The folders for pictures and tag sets are named img and gt, and the folders for training and validation sets are named training and validation. No test set list is generated.
python tools/create_dataset_list.py <your/dataset/dir> \
        --folder img gt --second_folder training validation
```
**Note:** A custom dataset directory must be specified, and FLAG can be set as needed. There is no need to specify `--type`.

- **For the cityscapes dataset**

If you are using the cityscapes dataset, you can run the following command to generate a file list.

```
# Generate a list of cityscapes files with a comma as the separator
python tools/create_dataset_list.py <your/dataset/dir> --type cityscapes --separator ","
```
**Note:**

The cityscapes dataset directory must be specified, and `--type` must be `cityscapes`.

Under the cityscapes type, part of the FLAG will be reset, no need to specify manually, as follows:

|FLAG|Fixed value|
|-|-|
|--folder|"leftImg8bit" "gtFine"|
|--format|"jpg" "png"|
|--postfix|"_leftImg8bit" "_gtFine_labelTrainIds"|

The remaining FLAG can be set as required.


After running, `train.txt`, `val.txt`, `test.txt` and `labels.txt` will be generated in the root directory of the dataset. PaddleSeg locates the image path by reading these text files.
