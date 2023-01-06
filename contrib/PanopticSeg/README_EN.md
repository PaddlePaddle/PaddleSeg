English | [简体中文](README_CN.md)

# Panoptic Segmentation Toolkit Based on PaddleSeg

## Contents

+ [Introduction](#introduction)
+ [Update Notes](#update-notes)
+ [Models](#models)
+ [Tutorials](#tutorials)
+ [Community](#community)

## Introduction

Panoptic segmentation is an image parsing task that unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). Built on top of [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), this toolkit aims to facilitate the training, evaluation, and deployment of panoptic segmentation models.

+ **High-Performance Models**: This toolkit provides state-of-the-art panoptic segmentation models that can be used out of the box.
+ **High Efficiency**: This toolkit supports multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of PaddlePaddle, all these allowing developers to train panoptic segmentation models at a lower cost.
+ **Complete Flow**: This toolkit supports a complete worflow from model designing to model deployment. With the help of this toolkit, developers can easily complete all tasks.

<p align="center">
<img src="https://user-images.githubusercontent.com/21275753/210925385-5021e2b6-2d73-4358-a9af-1e91cd9f008d.gif" height="150">
<img src="https://user-images.githubusercontent.com/21275753/210925394-57848331-0bd5-4c30-9fb0-03fc2a789936.gif" height="150">
<img src="https://user-images.githubusercontent.com/21275753/210925397-0b348fcf-b3f9-46cf-9512-b50278138658.gif" height="150">
</p>

+ *The pictures above are based on images from the [Cityscapes](https://www.cityscapes-dataset.com/) and [MS COCO](https://cocodataset.org/#home) datasets and the results obtained by models trained with this toolkit.*

## Update Notes

+ 2022.12
    - Add Mask2Former and Panoptic-DeepLab models.

## Models

+ [Mask2Former](configs/mask2former/README.md)
+ [Panoptic-DeepLab](configs/panoptic_deeplab/README.md)

## Tutorials
+ [Quick Start](docs/quick_start_en.md)
+ [Full Features](docs/full_features_en.md)
+ [Dev Guide](docs/dev_guide_en.md)

## Community

+ If you have any questions, suggestions or feature requests, please do not hesitate to create an issue in [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
+ Please scan the following QR code and join PaddleSeg WeChat group to communicate with us.

<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/174770518-e6b5319b-336f-45d9-9817-da12b1961fb1.jpg" width = "200" />  
</div>
