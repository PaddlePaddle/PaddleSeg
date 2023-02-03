English | [简体中文](cpp_inference_cn.md)
# Paddle Inference Deployment (C++)

## 1. Description

This document introduces an example of deploying a segmentation model on a Linux server (NV GPU or X86 CPU) using Paddle Inference's C++ interface. The main steps include:
* Prepare the environment
* Prepare models and pictures
* Compile and execute

PaddlePaddle provides multiple prediction engine deployment models (as shown in the figure below) for different scenarios. For details, please refer to [document](https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html).

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. Prepare the environment

### Prepare Paddle Inference C++ prediction library

You can download the Paddle Inference C++ prediction library from [link](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html).

Pay attention to select the exact version according to the machine's CUDA version, cudnn version, using MKLDNN or OpenBlas, whether to use TenorRT and other information. It is recommended to choose a prediction library with version >= 2.0.1.

Download the `paddle_inference.tgz` compressed file and decompress it, and save the decompressed paddle_inference file to `PaddleSeg/deploy/cpp/`.

If you need to compile the Paddle Inference C++ prediction library, you can refer to the [document](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html), which will not be repeated here.

### Prepare OpenCV

This example uses OpenCV to read images, so OpenCV needs to be prepared.

Run the following commands to download, compile, and install OpenCV.
````
sh install_opencv.sh
````

### Install Yaml, Gflags and Glog

This example uses Yaml, Gflags and Glog.

Run the following commands to download, compile, and install these libs.

````
sh install_yaml.sh
sh install_gflags.sh
sh install_glog.sh
````

## 3. Prepare models and pictures

Execute the following command in the `PaddleSeg/deploy/cpp/` directory to download the [test model](https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz) for testing. If you need to test other models, please refer to [documentation](../../model_export.md) to export the prediction model.

````
wget https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz
tar xf pp_liteseg_infer_model.tar.gz
````

Download one [image](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) from the validation set of cityscapes.

````
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
````

## 4. Compile and execute

Please check that `PaddleSeg/deploy/cpp/` stores prediction libraries, models, and pictures, as follows.

````
PaddleSeg/deploy/cpp
|-- paddle_inference # prediction library
|-- pp_liteseg_infer_model # model
|-- cityscapes_demo.png # image
````

Execute `sh run_seg_cpu.sh`, it will compile and then perform prediction on X86 CPU.

Execute `sh run_seg_gpu.sh`, it will compile and then perform prediction on Nvidia GPU.

The segmentation result will be saved in the "out_img.jpg" image in the current directory, as shown below. Note that this image is using histogram equalization for easy visualization.

![out_img](https://user-images.githubusercontent.com/52520497/131456277-260352b5-4047-46d5-a38f-c50bbcfb6fd0.jpg)
