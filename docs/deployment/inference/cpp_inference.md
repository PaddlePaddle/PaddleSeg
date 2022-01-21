# Paddle Inference部署（C++）

## 1. 说明

本文档介绍使用Paddle Inference的C++接口在Linux服务器端(NV GPU或者X86 CPU)部署分割模型的示例，主要步骤包括：
* 准备环境
* 准备模型和图片
* 编译、执行

飞桨针对不同场景，提供了多个预测引擎部署模型（如下图），详细信息请参考[文档](https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html)。

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

此外，PaddleX也提供了C++部署的示例和文档，具体参考[链接](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp)。

## 2. 准备环境

### 准备Paddle Inference C++预测库

大家可以从[链接](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载Paddle Inference C++预测库。

注意根据机器的CUDA版本、cudnn版本、使用MKLDNN或者OpenBlas、是否使用TenorRT等信息，选择准确版本。建议选择版本>=2.0.1的预测库。

下载`paddle_inference.tgz`压缩文件后进行解压，将解压的paddle_inference文件保存到`PaddleSeg/deploy/cpp/`下。

如果大家需要编译Paddle Inference C++预测库，可以参考[文档](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)，此处不再赘述。

### 准备OpenCV

本示例使用OpenCV读取图片，所以需要准备OpenCV。

执行如下命令下载、编译、安装OpenCV。
```
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xf 3.4.7.tar.gz

mkdir -p opencv-3.4.7/build
cd opencv-3.4.7/build

install_path=/usr/local/opencv3
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path} -DCMAKE_BUILD_TYPE=Release
make -j
make install

cd ../..
```

### 安装Yaml

本示例使用Yaml读取配置文件信息。

执行如下命令下载、编译、安装Yaml。

```
wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
unzip yaml-cpp-0.7.0.zip
mkdir -p yaml-cpp-yaml-cpp-0.7.0/build
cd yaml-cpp-yaml-cpp-0.7.0/build
cmake -DYAML_BUILD_SHARED_LIBS=ON ..
make -j
make install
```

## 3. 准备模型和图片

在`PaddleSeg/deploy/cpp/`目录下执行如下命令，下载[测试模型](https://paddleseg.bj.bcebos.com/dygraph/demo/stdc1seg_infer_model.tar.gz)用于测试。如果需要测试其他模型，请参考[文档](../../model_export.md)导出预测模型。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/stdc1seg_infer_model.tar.gz
tar xf stdc1seg_infer_model.tar.gz
```

下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

请检查`PaddleSeg/deploy/cpp/`下存放了预测库、模型、图片，如下。

```
PaddleSeg/deploy/cpp
|-- paddle_inference        # 预测库
|-- stdc1seg_infer_model    # 模型
|-- cityscapes_demo.png     # 图片
...
```

## 4. X86 CPU上部署

执行`sh run_seg_cpu.sh`，会进行编译，然后在X86 CPU上执行预测。

分割结果会保存在当前目录的“out_img.jpg“图片（如下图，该图片是使用了直方图均衡化，便于可视化）。

![out_img](https://user-images.githubusercontent.com/52520497/131456277-260352b5-4047-46d5-a38f-c50bbcfb6fd0.jpg)

## 5. Nvidia GPU上部署

PaddleInference支持两种方式在Nvidia GPU上部署模型：
* Naive方式：使用Paddle自实现的Kernel执行预测
* TRT方式：使用集成的TensorRT执行预测

所以，使用PaddleInference在Nvidia GPU上部署，有如下三种情况：
* 使用Naive形式对模型进行预测
* 使用TensorRT对固定shape的模型进行预测
* 使用TensorRT对动态shape的模型进行预测


如果使用“Naive形式对模型进行预测”，可以执行`sh run_seg_gpu.sh`，会进行编译、加载模型、加载图片、执行预测，结果保存在“out_img.jpg“图片。



Nvidia GPU上开启TensorRT部署

生成动态shape

执行`sh run_seg_gpu_trt_offline_dynamic_shape.sh`，会进行编译，然后在Nvidia GPU上执行预测。
