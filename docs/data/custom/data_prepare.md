English|[简体中文](data_prepare_cn.md)
# Custom Dataset

## 1、How to Use Datasets

We want to write the path of the image to the three folders `train.txt`, `val.txt`, `test.txt` and `labels.txt`, because PaddleSeg locates the image by reading these text files Path.
The texts of `train.txt`, `val.txt` and `test.txt` are divided into two columns with spaces as separators. The first column is the relative path of the image file relative to the dataset, and the second column is the relative path of the image file The relative path of the dataset. As follows:

```
images/xxx1.jpg (xx1.png) annotations/xxx1.png
images/xxx2.jpg (xx2.png) annotations/xxx2.png
...
```
`labels.txt`: Each line has a separate category, and the corresponding line number is the id corresponding to the category (line number starts from 0), as shown below:
```
labelA
labelB
...
```

## 2、Split Custom Dataset

We all know that the training process of neural network models is usually divided into training set, validation set, and test set. If you are using a custom dataset, PaddleSeg supports splitting the dataset by running scripts. If your dataset has been divided into the above three types, you can skip this step.

### 2.1 Original Image Requirements
The size of the original image data should be (h, w, channel), where h, w are the height and width of the image, and channel is the number of channels of the image.

### 2.2 Annotation Requirements
The annotated image must be a single-channel image, the annotated image should be in png format. The pixel value is the corresponding category, and the pixel annotated category needs to increase from 0.
For example, 0, 1, 2, 3 means that there are 4 categories, and the maximum number of labeled categories is 256. Among them, you can specify a specific pixel value to indicate that the pixel of that value does not participate in training and evaluation (the default is 255).

### 2.3 Spilit Custom Dataset and Generate File List

For all data that is not divided into training set, validation set, and test set, PaddleSeg provides a script to generate segmented data and generate a file list.

#### Use scripts to randomly split the custom dataset proportionally and generate a file list
The data file structure is as follows:
```
./dataset/  # Dataset root directory
|--images  # Original image catalog
|  |--xxx1.jpg (xx1.png)
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



## 3.Dataset file organization

* If you need to use a custom dataset for training, it is recommended to organize it into the following structure:
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

The contents of train.txt and val.txt are as follows:

    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...

If you only have a divided dataset, you can generate a file list by executing the following script:
```
# Generate a file list, the separator is a space, and the data format of the picture and the label set is png
python tools/create_dataset_list.py <your/dataset/dir> --separator " " --format png png
```
```
# Generate a list of files. The folders for pictures and tag sets are named img and gt, and the folders for training and validation sets are named training and validation. No test set list is generated.
python tools/create_dataset_list.py <your/dataset/dir> \
        --folder img gt --second_folder training validation
```
**Note:** A custom dataset directory must be specified, and FLAG can be set as needed. There is no need to specify `--type`.
After running, `train.txt`, `val.txt`, `test.txt` and `labels.txt` will be generated in the root directory of the dataset. PaddleSeg locates the image path by reading these text files.



* The labels of the annotated images are taken from 0, 1 in turn, and cannot be separated. If there are pixels that need to be ignored, they are labeled at 255.

The custom dataset can be configured as follows:
```yaml
train_dataset:
  type: Dataset
  dataset_root: custom_dataset
  train_path: custom_dataset/train.txt
  num_classes: 2
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: Normalize
  mode: train
```
Please pay attention to the storage location of **dataset path and training file**, according to the example of dataset_root and train_path in the code.
