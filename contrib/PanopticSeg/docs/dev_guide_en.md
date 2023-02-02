English | [简体中文](dev_guide_cn.md)

# Developer's Guide

This document introduces the advanced content of this toolkit and is helpful for readers who want to contribute their code.

## 0 Contents

+ [1 Directory Structure](#1-directory-structure)
+ [2 Data Structures](#2-data-structures)
+ [3 Encoding Panoptic Segmentation Labels](#3-encoding-panoptic-segmentation-labels)
+ [4 Dataset Customization](#4-dataset-customization)

## 1 Directory Structure

```plain
PanopticSeg
├── configs: Configuration files. Organized by model.
├── deploy: Code related to model deployment.
│   └── python: Python deployment code.
├── docs: Documents.
├── paddlepanseg:
│   ├── core: Core APIs that supports model training, evaluation, and inference.
│   ├── cvlibs: Core data structures.
│   │   ├── config.py: Manage configurations.
│   │   ├── info_dicts.py: Data containers for prediction results, samples, etc.
│   │   └── manager.py: Register and manage components.
│   ├── datasets: Interfaces for dataset IO.
│   ├── models: Implementations of models.
│   │   ├── backbones: Implementations of backbone networks.
│   │   ├── losses: Implementations of loss functions.
│   │   ├── ops: Implementations of external operators.
│   │   └── param_init.py: Utility functions for weight initialization.
│   ├── postprocessors: Postprocessors.
│   ├── transforms: Data transformation operators.
│   │   └── generate_targets: Transformation operators to generate training targets.
│   └── utils: Utility functions and classes.
├── test_tipc: TIPC scripts.
└── tools: Useful tools for users.
    ├── data: Tools for preparing public datasets.
    ├── analyze_model.py: Analyze the parameter count and computational complexity of a panoptic segmentation model.
    ├── export.py: Export model to static graph.
    ├── predict.py: Make inference with a pre-trained model and produce visualization results.
    ├── train.py: Train a model on a specific dataset.
    └── val.py: Evaluate the performance of a model on a specific dataset.
```

## 2 Data Structures

### 2.1 `InfoDict`

`InfoDict` objects are the containers for predictions, samples, etc. used in this toolkit. `InfoDict` is one of the most important data structures of this toolkit and does not exist in PaddleSeg. The `InfoDict` class is defined in `paddlepanseg/cvlibs/info_dicts.py`.

Basically we have four built-in `InfoDict` types: `SampleDict`, `NetOutDict`, `PPOutDict`, and `MetricDict`. They are all subclasses of `InfoDict`.

+ `SampleDict`: Designed to contain information and meta-information of data samples.
+ `NetOutDict`: Designed to contain network outputs.
+ `PPOutDict`: Designed to contain output of a postprocessor.
+ `MetricDict`: Designed to contain output of calling the `evaluate()` method of an `Evaluator` object.

The factory function `build_info_dict()` defined in `paddlepanseg/cvlibs/info_dicts.py` is the suggested way to construct an `InfoDict` object. By passing the `_type_` argument you specify the type of `InfoDict` (one of `'sample'`, `'net_out'`, `'pp_out'`, and `'metric'`).

## 3 Encoding Panoptic Segmentation Labels

Please find more details [here](encoding_protocol_en.md).

## 4 Dataset Customization

The recommended way of customizing your own dataset is to organize the dataset in [MS COCO format](https://cocodataset.org/#home) and write a new dataset interface inheriting from `paddlepanseg.datasets.base_dataset.COCOStylePanopticDataset`.

Adding a new subclass of `COCOStylePanopticDataset` usually involves the following steps:

+ Store the meta-information of the dataset in a list of dicts. See `CITYSCAPES_CATEGORIES` in `paddlepanseg/datasets/cityscapes.py` for an example.
+ Define a Python class that inherits from `paddlepanseg.datasets.base_dataset.COCOStylePanopticDataset`. Set the class attribute `CATEGORY_META_INFO` to the list of dicts you defined in the first step, and set the class attribute `NUM_CLASSES` to the number of classes in the dataset.
+ Rewrite the static method `_get_image_id()`, which accepts the path of the image and returns a unique identifier of that image.
+ Wrap the new dataset class with the decorator `paddlepanseg.cvlibs.manager.DATASETS.add_component()`. See `paddlepanseg/datasets/cityscapes.py` for an example.
