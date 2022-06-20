# Linux GPU/CPU C++ 推理功能测试

Linux GPU/CPU C++ 推理功能测试的主程序为`test_inference_cpp.sh`，可以测试基于C++预测引擎的推理功能。

## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU |
|  :----:   |  :----: |   :----:   |  :----:  |
|  STDC   |  stdc_stdc1 |  支持 | 支持 |
|  PP_LiteSeg   |  pp_liteseg_stdc1 |  支持 | 支持 |
|  PP_LiteSeg   |  pp_liteseg_stdc2 |  支持 | 支持 |
|  ConnectNet   |  pp_humanseg_lite |  支持 | 支持 |
|  HRNet W18 Small   | pp_humanseg_mobile  |  支持 | 支持 |
|  DeepLabV3P   |  pp_humanseg_server |  支持 | 支持 |
|  HRNet   |  fcn_hrnet_w18 |  支持 | 支持 |
|  OCRNet   |  ocrnet_hrnetw18 |  支持 | 支持 |
|  OCRNet   |  ocrnet_hrnetw48 |  支持 | 支持 |

## 2. 测试流程

### 2.1 准备数据和推理模型

#### 2.1.1 准备数据

在`PaddleSeg/test_tipc/cpp/`目录下执行如下命令，下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png)。

```
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
```

### 2.1.2 配置文件解析
完整的`{model_name}_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt`配置文件共有14行，包含两个方面的内容。
* 运行环境参数配置：第1~8行
* 模型参数配置：第10~14行

具体内容见[pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt](../configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt)

配置文件中主要有以下2种类型的字段。
* 一行内容以空格为分隔符：该行可以被解析为`key value`的格式，需要根据实际的含义修改该行内容，下面进行详细说明。
* 一行内容为`# xxxxx`：该行内容为注释信息，无需修改。
<details>
<summary><b>配置参数（点击以展开详细内容或者折叠）
</b></summary>

| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 2  | use_gpu      | 是否使用GPU    | 否         | 是           | value根据是否使用GPU进行修改               |
| 3  | gpu_id       | 使用的GPU卡号  | 否         | 是           | value修改为自己的GPU ID              |
| 4  | gpu_mem      | 显存          | 否         | 是           | value修改为自己的GPU 显存             |
| 5  | cpu_math_library_num_thread | 底层科学计算库所用线程的数量  | 否      | 是           | value修改为合适的线程数         |
| 6  | use_mkldnn   | 是否使用MKLDNN加速    | 否        | 是          | value根据是否使用MKLDNN进行修改          |
| 7  | use_tensorrt | 是否使用tensorRT进行加速          | 否         | 是           | value根据是否使用tensorRT进行修改             |
| 8  | use_fp16 | 是否使用半精度浮点数进行计算，该选项仅在use_tensorrt为true时有效 | 否         | 是          | value根据在开启tensorRT时是否使用半精度进行修改|
| 11 | model_path  | 预测模型结构文件路径         | 否         | 是           | value修改为预测模型结构文件路径 |
| 12 | params_path | 预测模型参数文件路径  | 否         | 是           | vvalue修改为预测模型参数文件路径 |
| 13 | is_resize  | 预处理时是否缩放图片         | 否         | 是           | value修改为是否缩放图像  
| 14 | is_normalize          | 预处理时是否做归一化    | 否         | 是           | value修改为是否归一化图像
| 15 | resize_width          | 预处理时图像裁剪后的宽度      | 否         | 是           | value修改为预处理时图像裁剪后的宽度
| 16 | resize_height          | 预处理时图像裁剪后的高度      | 否         | 是           | value修改为预处理时图像裁剪后的高度


</details>

#### 2.1.3 准备推理模型

本文开头列表中的模型均已支持通过prepare.sh自动下载进行测试。

如果需要测试其他模型，请参考[模型导出](../../docs/model_export_cn.md)导出预测模型。

请检查`PaddleSeg/test_tipc/cpp/`下存放了模型、图片，如下。

```
PaddleSeg/test_tipc/cpp/
|-- inference_models
    |-- stdc1seg_infer_model    # 模型
        |-- model.pdmodel
        |-- model.pdiparams
|-- cityscapes_demo.png     # 图片
|-- humanseg_demo.png       # 图片
...
```

**注意**：model.pdmodel、model.pdiparams的路径需要与[配置文件](../configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt)中的`model_path`和`params_path`参数对应一致。




### 2.2 准备环境

#### 2.2.1 运行准备

配置合适的编译和执行环境，其中包括编译器，cuda等一些基础库，建议安装docker环境，[参考链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

<details>
<summary><b>编译opencv、Paddle Inference、C++预测Demo（已写入prepare.sh自动执行，点击以展开详细内容或者折叠）
</b></summary>

#### 2.2.2 编译opencv库

* 首先需要从opencv官网上下载Linux环境下的源码，以3.4.7版本为例，下载及解压缩命令如下：

```
cd deploy/inference_cpp
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xvf 3.4.7.tar.gz
```

* 编译opencv，首先设置opencv源码路径(`root_path`)以及安装路径(`install_path`)，`root_path`为下载的opencv源码路径，`install_path`为opencv的安装路径。在本例中，源码路径即为当前目录下的`opencv-3.4.7/`。

```shell
cd ./opencv-3.4.7
export root_path=$PWD
export install_path=${root_path}/opencv3
```

* 然后在opencv源码路径下，按照下面的命令进行编译。

```shell
rm -rf build
mkdir build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON

make -j
make install
```

* `make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的代码编译。

以opencv3.4.7版本为例，最终在安装路径下的文件结构如下所示。**注意**：不同的opencv版本，下述的文件结构可能不同。

```
opencv3/
|-- bin     :可执行文件
|-- include :头文件
|-- lib64   :库文件
|-- share   :部分第三方库
```

#### 2.2.3 下载或者编译Paddle预测库

* 有2种方式获取Paddle预测库，下面进行详细介绍。

##### 直接下载安装

* [Paddle预测库官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)上提供了不同cuda版本的Linux预测库，可以在官网查看并选择合适的预测库版本。

  以`manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc8.2`版本为例，使用下述命令下载并解压：


```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz

tar -xvf paddle_inference.tgz
```

最终会在当前的文件夹中生成`paddle_inference/`的子文件夹,文件内容和上述的paddle_inference_install_dir一样。

##### 预测库源码编译
* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

* 进入Paddle目录后，使用如下命令编译。

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

更多编译参数选项可以参考Paddle C++预测库官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是之后进行C++预测时所需的Paddle库，`version.txt`中包含当前预测库的版本信息。


#### 2.2.4 编译C++预测Demo

* 编译命令如下，其中Paddle C++预测库、opencv等其他依赖库的地址需要换成自己机器上的实际地址。


```shell
cd test_tipc/cpp
sh build.sh
cd -
```

具体地，`build.sh`中内容大致如下。

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir
TENSORRT_DIR=your_tensorrt_lib_dir

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DDEMO_NAME=run_seg \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
```

上述命令中，

* `OPENCV_DIR`为opencv编译安装的地址（本例中为`opencv-3.4.7/opencv3`文件夹的路径）；

* `LIB_DIR`为下载的Paddle预测库（`paddle_inference`文件夹），或编译生成的Paddle预测库（`build/paddle_inference_install_dir`文件夹）的路径；

* `CUDA_LIB_DIR`为cuda库文件地址，在docker中一般为`/usr/local/cuda/lib64`；

* `CUDNN_LIB_DIR`为cudnn库文件地址，在docker中一般为`/usr/lib64`。

* `TENSORRT_DIR`是tensorrt库文件地址，在dokcer中一般为`/usr/local/TensorRT-7.2.3.4/`，TensorRT需要结合GPU使用。


以编译cpu
在执行上述命令，编译完成之后，会在当前路径下生成`build`文件夹，其中生成一个名为`seg_system`的可执行文件。

</details>

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_inference_cpp.sh ${your_params_file} ${your_infer_img_path}
```

以`pp_liteseg_stdc1`的为例命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt cpp_infer
bash test_tipc/test_inference_cpp.sh test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt
```



输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - pp_liteseg_stdc1 - ./test_tipc/cpp/build/seg_system test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt test_tipc/cpp/cityscapes_demo.png > ./test_tipc/output/infer_cpp/infer_cpp_use_cpu.log 2>&1 !
```

最终log中会打印出结果，如下所示
```
Current image path: ./test_tipc/cpp/cityscapes_demo.png
Current time cost: 0.567716 s, average time cost in all: 0.567716 s.
Finish, the result is saved in ./test_tipc/cpp/cityscapes_demo.png.jpg
```
详细log位于`./test_tipc/output/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log`中。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

预测图片包括预测标签图、经过直方图均衡化后的可视化效果图，分别保存在`PaddleSeg/output/cpp_predict`和`PaddleSeg/output/cpp_predict_vis`下。

可视化效果图展示：
![](./cityscapes_demo.jpg)
