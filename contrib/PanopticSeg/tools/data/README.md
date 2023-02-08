# Preparation of Public Datasets

In this directory we provide a couple of useful scripts for preprocessing of public datasets. Please read the following sections to understand the usage of each script.

## Cityscapes

1. Download Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/). The expected dataset structure is:

```plain
{PATH_TO_CITYSCAPES_DATASET}
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

2. Generate panoptic segmentation labels using `cityscapesscripts/preparation/createPanopticImgs.py` in [cityscapesScripts](https://github.com/mcordts/cityscapesScripts). For detailed usage, please refer to the [documentation](https://github.com/mcordts/cityscapesScripts/blob/master/README.md) of cityscapesScripts. Please note that **when processing the training subset, the `--use-train-id` should be specified**.

3. Switch to the root directory of this project, and run the following instructions to generate file lists:

```shell
python tools/data/create_cityscapes_file_lists.py --data_dir {PATH_TO_CITYSCAPES_DATASET} --out_dir {PATH_TO_CITYSCAPES_DATASET}
```

4. Make a symbolic link to the Cityscapes directory in `data/cityscapes`. Or you can choose to copy the entire directory. The expected dataset structure is:

```plain
data/cityscapes
├── gtFine
│   ├── cityscapes_panoptic_test
│   ├── cityscapes_panoptic_trainId
│   ├── cityscapes_panoptic_val
│   ├── test
│   ├── train
│   ├── val
│   ├── cityscapes_panoptic_test.json
│   ├── cityscapes_panoptic_train_trainId.json
│   └── cityscapes_panoptic_val.json
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── train_list.txt
└── val_list.txt
```

## MS COCO

1. Download MS COCO dataset from the [official website](https://cocodataset.org/#home). The expected dataset structure is:

```plain
{PATH_TO_COCO_DATASET}
├── annotations
│   ├── panoptic_train2017.json
│   └── panoptic_val2017.json
├── panoptic_train2017
├── panoptic_val2017
├── train2017
└── val2017
```

Note that we use the 2017 splits.

2. Switch to the root directory of this project, and run the following instructions to generate file lists:

```shell
python tools/data/create_coco_file_lists.py --data_dir {PATH_TO_COCO_DATASET} --out_dir {PATH_TO_COCO_DATASET}
```

4. Make a symbolic link to the COCO directory in `data/coco`. Or you can choose to copy the entire directory. The expected dataset structure is:

```plain
data/coco
├── annotations
│   ├── panoptic_train2017.json
│   └── panoptic_val2017.json
├── panoptic_train2017
├── panoptic_val2017
├── train2017
├── val2017
├── train_list.txt
└── val_list.txt
```
