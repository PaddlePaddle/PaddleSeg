# Dataset File Structure

This page describes the dataset file structures that PPLabel can import and export. **PPLabel may make modifications to files under the dataset folder.** Like during "Import Additional Data", new data files will be moved to this project's dataset path folder. Currently we won't delete anything. This behavior is intended to save disk space. **You should consider making a copy of the dataset as backup before import.** There will be a file named pplabel.warning under the dataset root folder that PPLabel is using. Avoid making changes to any file under the folder to avoid bugs.

PPLabel ships with sample datasets for each type of project. First create a sample dataset of any kind by clicking the "Sample Dataset" button on welcome page and then select a category. All sample datasets will be under ~/.pplabel/sample folder.

## Without Annotation

If the dataset doesn't contain any annotation, simply put all files under a single folder. PPLabel will walk through the folder (and all subfolders) to import all files it can annotate based on **file name extension**. All hidden files (whoses file name starts with .) will be ignored.

## Globally Supported Features

Dataset file structure varies across different types of projects but some features are supported in most types of project.

### labels.txt

labels.txt is supported in all project types not using COCO format annotation. PPLabel will look for a labels.txt file under the `Dataset Path` during import. You can list labels in this file, one for each line. For example:

```text
# labels.txt
Monkey
Mouse
```

PPLabel supports any string as label name. But label names may be used as folder names during dataset export, so avoid anything your os won't support like listed [here](https://stackoverflow.com/a/31976060). Other toolkits in the PaddlePaddle ecosystem, like [PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/data/format/classification.md), may also not support Chinese chracters as label names.

During import, labels.txt can contain more information than just label name. Currently, 4 formats are supported as listed below. | represents delimiter which defaults to space.

label length:

- 1: label name
- 2: label name | label id
- 3: label name | label id | hex color or common color name or grayscale value
- 5: label name | label id | r | g | b color

besides:

- //: string after // is stored as comment
- -: if you don't want to specify a label id but want to specify label color, put - in the label id field

Some examples:

```text
dog
monkey 4
mouse - #0000ff // mouse's id will be 5
cat 10 yellow
zibra 11 blue // some common colors are supported
snake 12 255 0 0 // rgb color
```

See [here](https://github.com/PaddleCV-SIG/PP-Label/blob/develop/pplabel/task/util/color.py#L15) for all supported color names.

During import, PPLabel will first create labels specified in labels.txt. So you are guarenteed the id for labels in this file will start from **0** and increase. During export this file will also be generated.

### xx_list.txt

xx_list.txt is supported in all project types not using COCO format annotation. xx_list.txt include `train_list.txt`, `val_list.txt` and `test_list.txt`. The files should be placed in the `Dataset Path` folder, same as labels.txt. These three files specify the dataset split and labels or data annotation file match (like for voc annotations, each line will be path to image file and path to annotation file) for each piece of data. File stucture for the three files are the same. Each line starts with path to a piece of data, relative to `Dataset Path`. It's followed by integers or strings indicating categories, or another path to annotation file. For example:

```text
# train_list.txt
image/9911.jpg 0 3
image/9932.jpg 4
image/9928.jpg Cat
```

For integers, PPLabel will look for the label in `labels.txt`, index starts from **0**. There can be multiple categories for one piece of data like in multi class image classification. To use a number as label name, you can either write the number down in `labels.txt` and provide label index in xx_list.txt. Or you can add a prefix to make it not a number like 10 -> n10. All three files will be generated during export, even when some of them are empty. Note that to ensure these files can be read by other toolkits in the PaddlePaddle ecosystem, datas having no annotation **won't** be included in `xx_list.txt`.

## Classification

PPLabel supports single class and multi class classification.

### Single Class Classification

Also know as ImageNet format. Sample datasets: [flowers102](https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip) [vegetables_cls](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)

Example Layout

```shell
Dataset Path
├── Cat
│   ├── cat-1.jpg
│   ├── cat-2.png
│   ├── cat-3.webp
│   └── ...
├── Dog
│   ├── dog-1.jpg
│   ├── dog-2.jpg
│   ├── dog-3.jpg
│   └── ...
├── monkey.jpg
├── train_list.txt
├── val_list.txt
├── test_list.txt
└── label.txt

# labels.txt
Monkey
Mouse
```

The folder name an image is in will be considered it's category. So the three cat and three dog images will have annotation after import. monkey.jpg won't have any annotation after import. Folder name labels will be created during import if they don't exist yet.

To avoid confilict, we only use dataset split information in the xx_list.txt file, **category information in these three files won't be considered**. You can use [this script](../tool/clas/mv_image_acc_split.py) to change the data's position accroding to the three xx_list.txt files before import.

### Multi Class Classification

In multi class classification, one piece of data can have multiple categories.

Example Layout

```shell
Dataset Path
├── image
│   ├── 9911.jpg
│   ├── 9932.jpg
│   └── monkey.jpg
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt

# labels.txt
cat
dog
yellow
black

# train_list.txt
image/9911.jpg 0 3
image/9932.jpg 4 0
image/9928.jpg monkey
```

In multi class classification, data's categories are only decided by xx_list.txt. Both label id and label name can be used. Folder names aren't considered.

## Detection

PPLabel supports two object detection dataset format: PASCAL VOC and COCO.

### PASCAL VOC

PASCAL VOC format stores annotations in xml files, one file for each image. Example Datasset: [Insect Detection](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)

Example Layout:

```shell
Dataset Path
├── Annotations
│   ├── 0001.xml
│   ├── 0002.xml
│   ├── 0003.xml
│   └── ...
├── JPEGImages
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   └── ...
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt
```

Format for the xml files is as follows

```text
<annotation>
 <folder>JPEGImages</folder>
 <filename></filename>
 <source>
  <database></database>
 </source>
 <size>
  <width></width>
  <height></height>
  <depth></depth>
 </size>
 <segmented>0</segmented>
 <object>
  <name></name>
  <pose></pose>
  <truncated></truncated>
  <difficult></difficult>
  <bndbox>
   <xmin></xmin>
   <ymin></ymin>
   <xmax></xmax>
   <ymax></ymax>
  </bndbox>
 </object>
</annotation>
```

In this format, we will treat all xml files under **Dataset Path** as annotations and match this annotation with image file at /Dataset Path/folder/filename. The folder and filename values are parsed from annotation xml. If folder node is not present in xml, the default value will be JPEGImages. If the folder node data is empty，image file should be at /Dataset Path/filename.

### COCO

COCO format keeps all information of a dataset in one file. We list part of COCO specifications below, please visit the [COCO website](https://cocodataset.org/#format-data) for more details. Note that in all projects using COCO format, xx_list.txt and labels.txt aren't supported. Example dataset: [Plane Detection](<>)

Example Layout:

```shell
Dataset Path
├── image
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   └── ...
├── train.json
├── val.json
└── test.json
```

COCO Format:

```text
{
    "info": info,
    "images": [image],
    "annotations": [annotation],
    "licenses": [license],
}

image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}


annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories[
{
 "id": int,
 "name": str,
 "supercategory": str,
 "color": str // this feature is specific to PPLabel. It's not in the coco spec.
}
]
```

We parse the annotation file with [pycocotoolse](https://github.com/linhandev/cocoapie). It's essentially the origional [pycocotools](https://github.com/cocodataset/cocoapi) with some dataset management features added. We look for three json files under the Dataset Path: `train.json`, `val.json` and `test.json`. Tasks parsed from these three files will go to the training, validation and test subset respectively. Be sure **not to define an image more than once across all files** otherwise import will fail. `xx_list.txt` and `labels.txt` aren't used in all projects using COCO format.

We will import all images under the `Dataset Path` folder as tasks. We match images on disk with image record in COCO json by looking for an image with relative path to `Dataset Path` ending with file_name value in COCO image record. For example an image with path `\Dataset Path\folder\image.png` will be match to image record with file_name `image.png`. If none or more than one match is found, import will fail. For example, images with path `\Dataset Path\folder1\image.png` and `\Dataset Path\folder2\image.png` will both be matched with image record with file_name value `image.png`. It's advised to put all images under a single folder to avoid duplicate image names.

If an image record doesn't have width or height, we will decide them by reading the image during import. This will slow down dataset import.

During export, the three json files will all be generated even if there is no image record in some of them.

In the categories section we added a color field. This field isn't in the origional coco spec. Color will be exported and used during import.

## Segmentation

We support two types of segmentation task and two dataset formats: semantic segmentation and instance segmentation task, mask and polygon format. Semantic and instance segmentation are the same with polygon format while mask format trests the two types of tasks differently.

### Polygon

For saving semantic or instance segmentation information as polygon we use the COCO format. The import and export process is virtually the same to [using COCO format with object detection project](#coco).

### MASK

In semantic segmentation, we only try to decide which category each pixel in the input image belongs to. The output will be a png image of the same size as the input image. Each pixel will be assigned a grayscale or color indicating a category.

Instance segmentation takes this one step further. We not only try to decide each pixel's category, but also differenciate between different instances of the same category. So each pixel will have two labels: it's cagtegory id and it's instance id.

### Semantic Segmentation

Example dataset: [optic disk segmentation](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz) (Note PPLabel cannot directly import this dataset. Masks in this dataset is in pesudo color. You have to modify the labels.txt file to specify the color for the optic disk class.)

Images and annotations are both image files in this format, so we placed more restrictions on the folder structure to tell them apart. We expect all images to be placed under `/Dataset Path/JPEGImages/` folder. All images under this folder will be imported, with or without annotation. Annotations should be placed in `/Dataset Path/Annotations`. Sample Layout:

```shell
Dataset Path
├── Annotations
│   ├── A0001.png
│   ├── B0001.png
│   ├── H0002.png
│   └── ...
├── JPEGImages
│   ├── A0001.jpg
│   ├── B0001.png
│   ├── H0002.bmp
│   └── ...
├── labels.txt
├── test_list.txt
├── train_list.txt
└── val_list.txt

# labels.txt
background -
optic_disk - 128 0 0 // for pesudo color mask, color for each label must be specified
```

During import, **in labels.txt, the first label will be treated as background and given label id 0**. For grayscale labels, we match the grayscale pixel value in masks with label id. For pesudo color labels, we match the color for each pixel with color specified in labels.txt. Import will fail if annotation doesn't have a matching label.

PNG is usually used for mask labels. We strip the file name extension from images and labels and match image to label with the same base file name. If multiple images with the same base file name plus different extension, like image.png and image.webp are found during import, import will fail.

During export, the first line of labels.txt will always be the background class. The values in mask images follow the same rule as during import. For grayscale masks, output will be a single channel image with label id as grayscale value. For pesodu color masks, output will be a three channel image with label color as color of each pixel.

### Instance Segmentation

The process of importing and exporting instance segmentation masks is similar to semantic segmentation. We store the masks as a two channel image in tiff format. The first channel (index 0) is label id, the second channel (index 1) is instance id.

[Napari](https://napari.org/#) is a convenient tool for inspecting tiff images. Install it following [official documentation](https://napari.org/#installation). Then:

- Open a image
  ![image](https://user-images.githubusercontent.com/29757093/178112182-1b7ae5d7-ab7b-4fee-b851-da2c43676da5.png)
- Open it's corresponding tiff mask PPLabel exports
  ![image](https://user-images.githubusercontent.com/29757093/178112188-e9c2e081-6752-4137-b60d-e64d9e7a11b6.png)
- Right click on the mask layer and select "Split Stack"
  ![image](https://user-images.githubusercontent.com/29757093/178112212-13c84d24-d753-4037-8851-d3e09f8fe9c8.png)
  ![image](https://user-images.githubusercontent.com/29757093/178112232-85feeec9-2ede-4045-9105-446b07454864.png)
- Right click on layer 0, select "Convert to Label" to see instance mask
  ![image](https://user-images.githubusercontent.com/29757093/178112305-6a0e36d2-3cab-4265-a88d-9ee55044b97e.png)
- Right click on layer 1, select "Convert to Label" to see category mask

## OCR

{
"image_name": \[
"transcription": "",
"illegibility": true/false, // is the line blurred
"points": \[\[w1, h1\], ..., \[wn, hn\]\], // w:horizontal, h:vertical
"language": "Latin" / "Chinese" / ...
\]
}
