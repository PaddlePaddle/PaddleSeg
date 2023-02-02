English | [简体中文](quick_start_cn.md)

# Quick Start

## 1 Installation

Please follow the instructions [here](./full_features_en.md#1-installation).

## 2 Making Predictions by Pre-trained Models

### 2.1 Download Pre-trained Weights and Demo Data

First, download pre-trained weights (`model.pdparams`) from [here](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/cityscapes/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k/model.pdparams). Then, acquire the image for demonstration (`demo.png`) [here](https://paddleseg.bj.bcebos.com/dygraph/panoptic_segmentation/tutorials/demo/demo.png). Put the model weights and the image in the root directory of this project.

### 2.2 Infer with Pre-trained Models

Execute the following instructions to make inference with the pre-trained model:

```shell
mkdir -p vis
python tools/predict.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k.yml \
    --model_path model.pdparams \
    --image_path demo.png \
    --save_dir vis
```

Note that we have used a dynamic-graph model for inference, which is usually of less efficiency. For more efficient model inference, please refer to the [deployment documentation](full_features_en.md#5-model-deployment).

### 2.3 See Visualization Results

There are three output images for each input image (with the same prefix), which can be found in `vis`.

The image `demo_sem.png` visualizes the network predictions from a semantic segmentation perspective:

<img src="https://user-images.githubusercontent.com/21275753/210925337-797befea-b774-4d63-849b-574709f098c7.png" height="300">

The image `demo_ins.png` visualizes the network predictions from an instance segmentation perspective:

<img src="https://user-images.githubusercontent.com/21275753/210925345-773f7c81-d281-4053-9684-6e8e6ac841f9.png" height="300">

The image `demo_pan.png` combines the first two images and visualizes the network predictions from a panoptic segmentation perspective:

<img src="https://user-images.githubusercontent.com/21275753/210925355-262775c2-3a9d-4c31-b45a-cef3bdebf4e0.png" height="300">

A detailed description of can be found [here](full_features_en.md#43-get-visualization-results).

## 3 Training and Evaluating Models

### 3.1 Prepare Datasets

This toolkit provides a few scripts to accommodate public panoptic segmentation datasets in order to use them for model training. For details of use, please see the [data preparation tutorial](../tools/data/README.md).

### 3.2 Train Models

First, find the model configuration file that you plan to use in `configs`, e.g. `configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k.yml`. Then, execute the following instructions:

```shell
python tools/train.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k.yml \
    --do_eval \
    --save_dir output
```

**Note that some models have prerequisites**. For example, you may have to compile an external C++/CUDA operator before performing any further steps. You can find the detailed instructions in the documentation for each model in `config`.

### 3.3 Evaluate Models

During and after training, you can find model checkpoints that store model weights and (possibly) optimizer parameters in `output` (specified by the `--save_dir` option in `tools/train.py`). You can evaluate the model weights that achieve the highest accuracy on the validation set (i.e. the checkpoint `output/best_model`) by running:

```shell
python tools/val.py \
    --config configs/panoptic_deeplab/panoptic_deeplab_resnet50_os32_cityscapes_1025x513_bs8_90k.yml \
    --model_path output/best_model/model.pdparams \
    --eval_sem \
    --eval_ins
```

By default the model is evaluated on the validation set, and the setting can be changed by modifying the configuration file. With `--eval_sem` and `--eval_ins`, semantic segmentation metrics (such as mIoU) and instance segmentation metrics (such as mAP) are also calculated and reported.
