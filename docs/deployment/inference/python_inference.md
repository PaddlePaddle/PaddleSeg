English|[简体中文](python_inference_cn.md)
# Paddle Inference Deployment（Python）

## 1. Description

This document introduces how to deploy the segmentation model on the server side (Nvidia GPU or X86 CPU) by Python api of Paddle Inference.

Paddle provides multiple prediction engine deployment models for different scenarios (as shown in the figure below), for more details, please refer to [document](https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html).

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. Prepare the model and data

Download [sample model](https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz) for testing.

If you want to use other models, please refer to [document](../../model_export.md) to export the model, and then test it.

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
```

Download a [picture](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) of cityscapes to test.

If the model is trained using other dataset, please prepare test images by yourself.

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

## 3. Prepare the deployment environment


Paddle Inference is the native inference library of Paddle, which provides server-side deployment model. Using the Python interface to deploy Paddle Inference model, you need to install PaddlePaddle according to the deployment situation. That is, the Python interface of Paddle Inference is integrated in PaddlePaddle.

On the server side, Paddle Inference models can be deployed on Nvidia GPU or X86 CPU. The model deployed on Nvidia GPU has fast calculation speed, and on X86 CPU has a wide range of applications.



1) Prepare the X86 CPU deployment environment

If you deploy the model on X86 CPU, please refer to the [document](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to prepare the environment and install the CPU version of PaddlePaddle (recommended version>=2.1)，Read the installation document in detail, and choose to install the correct version of PaddlePaddle according to whether the X86 CPU machine supports avx instructions.

2) Prepare Nvidia GPU deployment environment

Paddle Inference deploys the model on the Nvidia GPU side and supports two calculation methods: Naive method and TensorRT method. The TensorRT method has a variety of calculation accuracy, which is usually faster than the Naive method.

If you use the Naive method to deploy the model on the Nvidia GPU, you can refer to the [document](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to prepare the CUDA environment and install the corresponding GPU version of PaddlePaddle（recoment paddlepaddle-gpu>=2.1. For example:

```
# CUDA10.1 PaddlePaddle
python -m pip install paddlepaddle-gpu==2.1.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

If you use TensorRT to deploy the model on the Nvidia GPU, please refer to the [document](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to prepare the CUDA environment and install the corresponding GPU version of PaddlePaddle（recoment paddlepaddle-gpu>=2.1. For example:

```
python -m pip install paddlepaddle-gpu==[version] -f https://www.paddlepaddle.org.cn/whl/stable/tensorrt.html
```

To deploy the model using TensorRT on Nvidia GPU, you need to download the TensorRT library. 
The CUDA10.1+cudnn7 environment requires TensorRT 6.0, and the CUDA10.2+cudnn8.1 environment requires TensorRT 7.1. You can download it on the [TensorRT official website](https://developer.nvidia.com/tensorrt). We only provide the link of TensorRT under Ubuntu system here.

```
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.1-cudnn7.6-trt6.0.tar
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.2-cudnn8.0-trt7.1.tgz
```

Download and decompress the TensorRT library, and add the path of the TensorRT library to LD_LIBRARY_PATH, `export LD_LIBRARY_PATH=/path/to/tensorrt/:${LD_LIBRARY_PATH}`


## 4. Inference

In the root directory of PaddleSeg, execute the following command to predict:

```shell
python deploy/python/infer.py \
    --config /path/to/model/deploy.yaml \
    --image_path /path/to/image/path/or/dir
```

**The parameter description is as follows:**
|Parameter name|Function|Is it a required option|Default|
|-|-|-|-|
|config|**The configuration file generated when exporting the model**, ot the configuration file in the configs directory|Yes|-|
|image_path|The path or directory or file list of the input picture|Yes|-|
|batch_size|Batch size for single card |No|1|
|save_dir|Path to save result|No|output|
|device|Inference device, options are'cpu','gpu'|No|'gpu'|
|use_trt|Whether to enable TensorRT to accelerate prediction \(when device=gpu, this parameter takes effect\)|No|False|
|precision|The precision when enable TensorRT, the options are'fp32','fp16','int8' \(when device=gpu, this parameter takes effect\)|No|'fp32'|
|enable_auto_tune|When Auto Tune is turned on, part of the test data will be collected dynamic shapes offline for TRT deployment \(this parameter take effect when device=gpu, use_trt=True, and paddle version>=2.2\)|No| False |
|cpu_threads|The number of cpu threads \(when device=cpu, this parameter takes effect\)|No|10|
|enable_mkldnn|whether to use MKL-DNN to accelerate cpu prediction \(when device=cpu, this parameter takes effect\)|No|False|
|benchmark|Whether to generate logs, including environment, model, configuration, and performance information|No|False|
|with_argmax|Perform argmax operation on the prediction result|No|False|

**Instructions**
* If you deploy the model on an X86 CPU, you must set the device to cpu. In addition, the unique parameters for CPU deployment include cpu_threads and enable_mkldnn.
* If you use Naive mode to deploy the model on the Nvidia GPU, you must set the device to gpu.
* If you use TensorRT to deploy the model on Nvidia GPU, you must set device to gpu and use_trt to True. This method supports three precisions:
    * Load the conventional prediction model, set precision to fp32, and execute fp32 precision at this time
    * Load the conventional prediction model, set the precision to fp16, and execute the fp16 precision. It can speed up the inference time
    * Load the quantitative prediction model, set precision to int8, and execute int8 precision. It can speed up the  inference time
* If you use TensorRT mode to deploy the model on the Nvidia GPU and appears an error message `(InvalidArgument) some trt inputs dynamic shape inof not set`, you can set the enable_auto_tune parameter as True. At this time, use part of the test data to collect dynamic shapes offline, and use the collected dynamic shapes for TRT deployment. (Note that a small number of models do not support deployment using TensorRT on Nvidia GPUs).
* If you want to enable it --benchmark, you need to install auto_log, please refer to [installation method](https://github.com/LDOUBLEV/AutoLog).

The prediction results of the test sample are as follows:

![cityscape_predict_demo.png](../../images/cityscapes_predict_demo.png)

