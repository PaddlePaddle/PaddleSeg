简体中文 | [English](cpp_inference.md)
# Paddle Inference部署（C++）

## 1. 说明

本文档介绍使用Paddle Inference的C++接口在Linux服务器端(NV GPU或者X86 CPU)部署分割模型的示例，主要步骤包括：
* 准备环境
* 准备模型和图片
* 编译、执行

飞桨针对不同场景，提供了多个预测引擎部署模型（如下图），详细使用方法请参考[文档](https://www.paddlepaddle.org.cn/inference/v2.3/product_introduction/summary.html)。

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. 准备环境
### 2.1 准备基础环境

如果在X86 CPU上部署模型，不需要下面CUDA、cudnn、TensorRT的准备工作。

如果在Nvidia GPU上部署模型，必须安装必CUDA、cudnn。此外，PaddleInference在Nvidia GPU上支持使用TensorRT进行加速，可以视需要安装。

此处，我们提供两个版本的CUDA、cudnn、TensorRT文件下载。
```
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.1-cudnn7.6-trt6.0.tar
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.2-cudnn8.0-trt7.1.tgz
```

下载解压后，CUDA和cudnn可以参考网上文档或者官方文档([Cuda Doc](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [cudnn Doc](https://docs.nvidia.com/deeplearning/cudnn/install-guide/))进行安装。TensorRT只需要设置库路径，比如：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/TensorRT-7.1.3.4/lib
```

如果大家使用Docker，可以拉取`registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`，在docker内部配置需要的基础环境。

### 2.2 准备Paddle Inference C++预测库

如果在X86 CPU上部署模型，进入[C++预测库](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html)下载“manylinux_cpu_xxx”命名的PaddleInference C++预测库。

如果在Nvidia GPU上部署模型，进入[C++预测库](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html)下载对应CUDA、Cudnn、TRT、GCC版本的PaddleInference C++预测库。


> 不同C++预测库可以根据名字进行区分。请根据机器的操作系统、CUDA版本、cudnn版本、使用MKLDNN或者OpenBlas、是否使用TenorRT、GCC版本等信息，选择准确版本。（建议选择版本>=2.3的预测库）

下载`paddle_inference.tgz`压缩文件后进行解压，将解压的paddle_inference文件保存到`PaddleSeg/deploy/cpp/`下。

如果大家需要编译Paddle Inference C++预测库，可以参考[文档](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html)，此处不再赘述。

### 2.3 安装其他库

本示例使用OpenCV读取图片，所以需要安装OpenCV。在实际部署中，大家视需要安装。

执行如下命令下载、编译、安装OpenCV。

```
sh install_opencv.sh
```

本示例使用Yaml读取配置文件信息，使用Gflags和Glog管理输入和输出。在实际部署中，大家视需要安装。

```
sh install_yaml.sh
sh install_gflags.sh
sh install_glog.sh
```

## 3. 准备模型和图片

在`PaddleSeg/deploy/cpp/`目录下执行如下命令，下载[测试模型](https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz)。如果需要测试其他模型，请参考[文档](../../model_export.md)导出预测模型。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz
tar xf pp_liteseg_infer_model.tar.gz
```

预测模型文件格式如下。
```shell
output/inference_model
  ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理方式等信息
  ├── model.pdmodel          # 预测模型的拓扑结构文件
  ├── model.pdiparams        # 预测模型的权重文件
  └── model.pdiparams.info   # 参数额外信息，一般无需关注
```

`model.pdmodel`可以通过[Netron](https://netron.app/)打开进行模型可视化，大家可以看到预测模型的输入输出的个数、数据类型（比如int32_t, int64_t, float等）。
如果模型的输出数据类型不是int32_t，执行默认的代码后会报错。此时需要大家手动修改`deploy/cpp/src/test_seg.cc`文件中的下面代码，改为输出对应的数据类别。
```
std::vector<int32_t> out_data(out_num);
```

下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

请检查`PaddleSeg/deploy/cpp/`下存放了预测库、模型、图片，如下。

```
PaddleSeg/deploy/cpp
|-- paddle_inference        # 预测库
|-- pp_liteseg_infer_model    # 模型
|-- cityscapes_demo.png     # 图片
...
```

## 4. X86 CPU上部署

执行`sh run_seg_cpu.sh`，会进行编译，然后在X86 CPU上执行预测，分割结果会保存在当前目录的“out_img.jpg“图片。

## 5. Nvidia GPU上部署

在Nvidia GPU上部署模型，我们需要提前明确部署场景和要求，主要关注多次预测时输入图像的尺寸是否变化。

定义：固定shape模式是指多次预测时输入图像的尺寸是不变的，动态shape模式是指每次预测时输入图像的尺寸可以变化。

飞桨PaddleInference在Nvidia GPU上部署模型，支持两种方式：
* Naive方式：使用Paddle自实现的Kernel执行预测；它使用相同的配置方法支持固定shape模式和动态shape模式。
* TRT方式：使用集成的TensorRT执行预测，通常TRT方式比Naive方式速度更快；它使用不同的配置方法支持固定shape模式和动态shape模式。

### 5.1 Naive方式-部署

如果使用Naive方式部署Seg分割模型（固定Shape模式或者动态Shape模式），可以执行`sh run_seg_gpu.sh`。

该脚本会进行编译、加载模型、加载图片、执行预测，结果保存在“out_img.jpg“图片。

### 5.2 TRT方式-固定Shape模式-部署

使用TRT方式、固定Shape模式来部署PaddleSeg分割模型：
* 打开`run_seg_gpu_trt.sh`脚本，设置`TENSORRT_ROOT`为机器中TensorRT库的路径，比如`TENSORRT_ROOT='/work/TensorRT-7.1.3.4/'`。
* 执行 `sh run_seg_gpu_trt.sh`。
* 预测结果会保存在“out_img.jpg“图片。

对于PaddleSeg分割模型，通常是支持任意输入size，模型内部存在动态Shape的OP。
所以使用TRT方式、固定Shape模式来部署时，经常会出现错误。这时就推荐使用TRT方式、动态Shape模式来部署。

### 5.3 TRT方式-动态Shape模式-部署

PaddleInference有多种方法使用TRT方式、固定Shape模式来部署PaddleSeg分割模型，此处推荐一种通用性较强的方法，主要步骤包括：准备预测模型和样本图像；离线收集动态Shape；部署执行。

* 准备预测模型和样本图像

准备预测模型和样本图像，是用于离线收集动态Shape，所以**准备的样本图像需要包含实际预测时会遇到的最大和最小图像尺寸**。

在前面步骤，我们已经准备好预测模型和一张测试图片。

* 离线收集动态Shape

请参考PaddleSeg[安装文档](../../install_cn.md)安装PaddlePaddle和PaddleSeg的依赖项。

在`PaddleSeg/deploy/cpp`路径下，执行如下命令。
```
python ../python/collect_dynamic_shape.py \
    --config pp_liteseg_infer_model/deploy.yaml \
    --image_path ./cityscapes_demo.png \
    --dynamic_shape_path ./dynamic_shape.pbtxt
```

通过指定预测模型config文件和样本图像，脚本会加载模型、读取样本图像、统计并保存动态Shape到`./dynamic_shape.pbtxt`文件。

如果有多张样本图像，可以通过`--image_path`指定图像文件夹。

* 部署执行

打开`run_seg_gpu_trt_dynamic_shape.sh`脚本，设置`TENSORRT_ROOT`为机器上TensorRT库的路径，设置`DYNAMIC_SHAPE_PATH`为动态Shape文件。

执行`sh run_seg_gpu_trt_dynamic_shape.sh`，预测结果会保存在“out_img.jpg“图片。

结果如下图，该图片使用了直方图均衡化，便于可视化。

![out_img](https://user-images.githubusercontent.com/52520497/131456277-260352b5-4047-46d5-a38f-c50bbcfb6fd0.jpg)
