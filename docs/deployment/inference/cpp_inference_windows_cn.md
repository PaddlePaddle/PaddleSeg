简体中文 | [English](cpp_inference_windows.md)

# Paddle Inference在Windows上部署（C++）

## 1. 说明

本文档介绍使用Paddle Inference的C++接口在Windows 10部署语义分割模型的示例，主要步骤包括：
* 准备环境
* 准备模型和图片
* 编译、执行

飞桨针对不同场景，提供了多个预测引擎部署模型（如下图），详细使用方法请参考[文档](https://www.paddlepaddle.org.cn/inference/v2.3/product_introduction/summary.html)。

![inference_ecosystem](https://user-images.githubusercontent.com/52520497/130720374-26947102-93ec-41e2-8207-38081dcc27aa.png)

## 2. 准备环境

### 2.1 准备基础环境

模型部署的基础环境要求如下：
* Visual Studio 2019 (根据Paddle预测库所使用的VS版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=vs-2019) )
* CUDA / CUDNN / TensorRT (仅在使用GPU版本的预测库时需要)
* CMake 3.0+ [CMake下载](https://cmake.org/download/)

下面所有示例以工作目录为`D:\projects`进行演示。

### 2.2 准备 CUDA/CUDNN/TensorRT 环境

模型部署的环境和需要准备的库对应如下表：

|  部署环境   |          库     |
|:-------:|:-------------------:|
|   CPU   |          -          |
|   GPU   |     CUDA/CUDNN      |
| GPU_TRT | CUDA/CUDNN/TensorRT |

使用GPU进行推理的用户需要参考如下说明准备CUDA和CUDNN，使用CPU推理的用户可以跳过。  

CUDA安装请参考[官方教程](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-cuda-enabled-system)。  
CUDA的默认安装路径为`C:\Program Files\NVIDIA GPU Computing Toolkit`，将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y\bin`添加到环境变量中。

CUDNN安装请参考[官方教程](https://docs.nvidia.com/deeplearning/cudnn/install-guide/#install-windows)。  
将cudnn的`bin`、`include`、`lib`文件夹内的文件复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y`的`bin`、`include`、`lib`文件夹。（Vx.y中的x.y表示cuda版本）  

如果在CUDA下使用TensorRT进行推理加速，还需要准备TensorRT，具体请参考[官方教程](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)。  
将安装目录`lib`文件夹的`.dll`文件复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\Vx.y\bin`。

### 2.3 准备Paddle Inference C++预测库
Paddle Inference C++ 预测库针对不同的CPU和CUDA版本提供了不同的预编译版本，大家根据自己的环境选择合适的预编译库：[C++预测库下载链接](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows) 。

如果提供的预编译库不满足需求，可以自己编译Paddle Inference C++预测库，参考[文档](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html) ，此处不再赘述。

本文档以CUDA=11.6，CUDNN=8.4.1.5，TensorRT=8.4.1.5为例进行介绍。

Paddle Inference目录结构：
```shell
D:\projects\paddle_inference
  ├── paddle
  ├── third_party
  ├── CMakeCache.txt
  └── version.txt
```

### 2.4 安装OpenCV
本示例使用OpenCV读取图片，所以需要安装OpenCV。在其他的项目中，大家视需要安装。

1. 在OpenCV官网下载适用于Windows平台的4.6.0版本，[下载地址](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download)  
2. 运行下载的可执行文件，将OpenCV解压至指定目录，如`D:\projects\opencv`
3. 配置环境变量，如下流程所示（如果使用全局绝对路径，可以不用设置环境变量）  
    - `我的电脑`->`属性`->`高级系统设置`->`环境变量`
    - 在系统变量中找到`Path`（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc15\bin`


## 3. 准备模型和图片

大家可以下载准备好的[预测模型](https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz)到本地，用于后续测试。
如果需要测试其他模型，请参考[文档](../../model_export.md)导出预测模型。

预测模型文件格式如下：
```shell
pp_liteseg_infer_model
  ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理方式等信息
  ├── model.pdmodel          # 预测模型的拓扑结构文件
  ├── model.pdiparams        # 预测模型的权重文件
  └── model.pdiparams.info   # 参数额外信息，一般无需关注
```

`model.pdmodel`可以通过[Netron](https://netron.app/) 打开进行模型可视化，点击输入节点即可看到预测模型的输入输出的个数、数据类型（比如int32_t, int64_t, float等）。
如果模型的输出数据类型不是int32_t，执行默认的代码后会报错。此时需要大家手动修改`deploy/cpp/src/test_seg.cc`文件中的下面代码，改为输出对应的数据类别：
```
std::vector<int32_t> out_data(out_num);
```

下载cityscapes验证集中的一张[图片](https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png) 到本地，用于后续测试。

## 4. 编译

工程整体目录结构如下：
```shell
D:\projects
  ├── opencv
  ├── paddle_inference
  └── PaddleSeg
```


### 4.1 使用CMake生成项目文件

编译参数的说明如下，其中带`*`表示仅在使用**GPU版本**预测库时指定，带`#`表示仅在使用**TensorRT**时指定。

| 参数名              | 含义                               |
|------------------|----------------------------------|
| *WITH_GPU        | 是否使用GPU，默认为OFF；                  |
| *CUDA_LIB        | CUDA的库路径；                        |
| *USE_TENSORRT    | 是否使用TensorRT，默认为OFF；             |
| #TENSORRT_DLL    | TensorRT的.dll文件存放路径；             |
| WITH_MKL         | 是否使用MKL，默认为ON，表示使用MKL，若设为OFF，则表示使用Openblas； |
| CMAKE_BUILD_TYPE | 指定编译时使用Release或Debug；            |
| PADDLE_LIB_NAME  | Paddle预测库名称；                     |
| OPENCV_DIR       | OpenCV的安装路径；                     |
| PADDLE_LIB       | Paddle预测库的安装路径；                  |
| DEMO_NAME        | 可执行文件名；                          |


进入`cpp`目录下：
```
cd D:\projects\PaddleSeg\deploy\cpp
```

创建`build`文件夹，并进入其目录：
```commandline
mkdir build
cd build
```

执行编译命令的格式如下：

(**注意**：若路径中包含空格，需使用引号括起来。)
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=ON -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB=path_to_cuda_lib -DOPENCV_DIR=path_to_opencv -DPADDLE_LIB=path_to_paddle_dir -DTENSORRT_DLL=path_to_tensorrt_.dll -DDEMO_NAME=test_seg
```

例如，GPU不使用TensorRT推理，命令如下：
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=OFF -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

GPU使用TensorRT推理，命令如下：
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DUSE_TENSORRT=ON -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DCUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DTENSORRT_DLL="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin" -DDEMO_NAME=test_seg
```

CPU使用MKL推理，命令如下：
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

CPU使用OpenBlas推理，命令如下：
```
cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\opencv -DPADDLE_LIB=D:\projects\paddle_inference -DDEMO_NAME=test_seg
```

### 4.2 编译

用`Visual Studio 2019`打开`cpp\build\cpp_inference_demo.sln`，将编译模式设置为`Release`，点击`生成`->`生成解决方案`，在`cpp\build\Release`文件夹内生成`test_seg.exe`。

## 5、执行

进入到`build/Release`目录下，将准备的模型和图片放到`test_seg.exe`同级目录，`build/Release`目录结构如下：
```shell
Release
├──test_seg.exe                # 可执行文件
├──cityscapes_demo.png         # 测试图片
├──pp_liteseg_infer_model      # 推理用到的模型
    ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理方式等信息
    ├── model.pdmodel          # 预测模型的拓扑结构文件
    ├── model.pdiparams        # 预测模型的权重文件
    └── model.pdiparams.info   # 参数额外信息，一般无需关注
├──*.dll                       # dll文件
```

运行以下命令进行推理，GPU推理：
```commandline
test_seg.exe --model_dir=./pp_liteseg_infer_model --img_path=./cityscapes_demo.png --devices=GPU
```

CPU推理：
```commandline
test_seg.exe --model_dir=./pp_liteseg_infer_model --img_path=./cityscapes_demo.png --devices=CPU
```

预测结果保存为`out_img.jpg`，该图片使用了直方图均衡化，便于可视化，如下图：

![out_img](https://user-images.githubusercontent.com/52520497/131456277-260352b5-4047-46d5-a38f-c50bbcfb6fd0.jpg)
