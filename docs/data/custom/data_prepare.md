English | [简体中文](data_prepare_cn.md)
# Prepare Custom Dataset Data
If you want to train on custom dataset, please prepare the dataset using following steps.

## 1. Split files and generate required txts.
We suggest that you split your dataset into the following structure:

```
    custom_dataset
        |
        |--images
        |  |--image1.jpg
        |  |--image2.jpg
        |  |--...
        |
        |--labels
        |  |--label1.jpg
        |  |--label2.png
        |  |--...
        |
        |--train.txt
        |
        |--val.txt
        |
        |--test.txt
```

The contents of train.txt and val.txt are as follows. They are used to load data in [dataset class](../../../paddleseg/datasets/dataset.py), which is the parent class of all dataset.
```
    images/image1.jpg labels/label1.png
    images/image2.jpg labels/label2.png
    ...
```

### 1.1 Split your data
If your dataset is not organized as the aforementioned structure, we suggest that you refer to the following script to generate the corresponding file structure and txt files. Firstly, we need to organize our files as the following structure, whereas the filenames are free to change :

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

The commands used are as follows, which supports enabling specific functions through different Flags.
```
python tools/data/split_dataset_list.py <dataset_root> <images_dir_name> <labels_dir_name> ${FLAGS}
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
python tools/data/split_dataset_list.py <dataset_root> images annotations --split 0.6 0.2 0.2 --format jpg png
```

### 1.2 Generate txt files
If you only have a divided dataset, you can generate a file list by executing the following script:
```
# Generate a file list, the separator is a space, and the data format of the picture and the label set is png
python tools/data/create_dataset_list.py <your/dataset/dir> --separator " " --format png png
```
```
# Generate a list of files. The folders for pictures and tag sets are named img and gt, and the folders for training and validation sets are named training and validation. No test set list is generated.
python tools/data/create_dataset_list.py <your/dataset/dir> \
        --folder img gt --second_folder training validation
```
**Note:** A custom dataset directory must be specified, and FLAG can be set as needed. There is no need to specify `--type`.
After running, `train.txt`, `val.txt`, `test.txt` and `labels.txt` will be generated in the root directory of the dataset. PaddleSeg locates the image path by reading these text files.

* The labels of the annotated images are taken from 0, 1 in turn, and cannot be separated. If there are pixels that need to be ignored, they are labeled at 255.

## 2. Config your dataset
The custom dataset can be configured as follows:
```yaml
train_dataset:
  type: Dataset
  dataset_root: the-relative-path-to-your-data
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
