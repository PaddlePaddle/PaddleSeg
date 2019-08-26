# PaddleSeg C++预测部署方案

## 说明

本目录提供一个跨平台的图像分割模型的C++预测部署方案，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

主要设计的目标包括以下三点：
- 跨平台，支持在 windows和Linux完成编译、开发和部署
- 支持主流图像分割任务，用户通过少量配置即可加载模型完成常见预测任务，比如人像分割等
- 可扩展性，支持用户针对新模型开发自己特殊的数据预处理、后处理等逻辑



## 主要目录和文件
| 文件 | 作用 |
|-------|----------|
| CMakeList.txt | cmake 编译配置文件 |
| external-cmake| 依赖的外部项目 cmake (目前仅有yaml-cpp)|
| demo.cpp | 示例C++代码，演示加载模型完成预测任务 |
| predictor | 加载模型并预测的类代码|
| preprocess |数据预处理相关的类代码|
| utils | 一些基础公共函数|
| images/humanseg | 样例人像分割模型的测试图片目录|
| conf/humanseg.yaml | 示例人像分割模型配置|
| tools/visualize.py | 预测结果彩色可视化脚本 |

## Windows平台编译

### 前置条件
* Visual Studio 2015+ 
* CUDA 8.0 / CUDA 9.0 + CuDNN 7
* CMake 3.0+

我们分别在 `Visual Studio 2015` 和 `Visual Studio 2019 Community` 两个版本下做了测试.

**下面所有示例，以根目录为 `D:\`演示**

### Step1: 下载代码

1. `git clone http://gitlab.baidu.com/Paddle/PaddleSeg.git`
2. 拷贝 `D:\PaddleSeg\inference\` 目录到 `D:\PaddleDeploy`下

目录`D:\PaddleDeploy\inference` 目录包含了`CMakelist.txt`以及代码等项目文件.



### Step2: 下载PaddlePaddle预测库fluid_inference

根据Windows环境，下载相应版本的PaddlePaddle预测库，并解压到`D:\PaddleDeploy\`目录

| CUDA | GPU | 下载地址 |
|------|------|--------|
| 8.0 | Yes | [fluid_inference.zip](https://bj.bcebos.com/v1/paddleseg/fluid_inference_win.zip) |
| 9.0 | Yes | [fluid_inference_cuda90.zip](https://paddleseg.bj.bcebos.com/fluid_inference_cuda9_cudnn7.zip) |

`D:\PaddleDeploy\fluid_inference`目录包含内容为：
```bash
paddle # paddle核心目录
third_party # paddle 第三方依赖
version.txt # 编译的版本信息
```


### Step3: 安装配置OpenCV

1. 在OpenCV官网下载适用于Windows平台的3.4.6版本， [下载地址](https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download)  
2. 运行下载的可执行文件，将OpenCV解压至指定目录，如`D:\PaddleDeploy\opencv`  
3. 配置环境变量，如下流程所示  
    1. 我的电脑->属性->高级系统设置->环境变量  
    2. 在系统变量中找到Path（如没有，自行创建），并双击编辑  
    3. 新建，将opencv路径填入并保存，如`D:\PaddleDeploy\opencv\build\x64\vc14\bin` 

### Step4: 以VS2015为例编译代码

以下命令需根据自己系统中各相关依赖的路径进行修改

* 调用VS2015, 请根据实际VS安装路径进行调整，打开cmd命令行工具执行以下命令
* 其他vs版本，请查找到对应版本的`vcvarsall.bat`路径，替换本命令即可

```
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
```
    
* CMAKE编译工程
    * PADDLE_DIR: fluid_inference预测库目录
    * CUDA_LIB: CUDA动态库目录, 请根据实际安装情况调整
    * OPENCV_DIR: OpenCV解压目录

```
# 创建CMake的build目录
D:
cd PaddleDeploy\inference
mkdir build
cd build
D:\PaddleDeploy\inference\build> cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=ON -DPADDLE_DIR=D:\PaddleDeploy\fluid_inference -DCUDA_LIB=D:\PaddleDeploy\cudalib\v8.0\lib\x64 -DOPENCV_DIR=D:\PaddleDeploy\opencv -T host=x64
```

这里的`cmake`参数`-G`, 可以根据自己的`VS`版本调整，具体请参考[cmake文档](https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html)

* 生成可执行文件

```
D:\PaddleDeploy\inference\build> msbuild /m /p:Configuration=Release cpp_inference_demo.sln
```

### Step5: 预测及可视化

上步骤中编译生成的可执行文件和相关动态链接库并保存在build/Release目录下，可通过Windows命令行直接调用。
可下载并解压示例模型进行测试，点击下载示例的人像分割模型[下载地址](https://paddleseg.bj.bcebos.com/inference_model/deeplabv3p_xception65_humanseg.tgz)

假设解压至 `D:\PaddleDeploy\models\deeplabv3p_xception65_humanseg` ，执行以下命令：

```
cd Release
D:\PaddleDeploy\inference\build\Release> demo.ext --conf=D:\\PaddleDeploy\\inference\\conf\\humanseg.yaml --input_dir=D:\\PaddleDeploy\\inference\\images\humanseg\\
```

预测使用的两个命令参数说明如下：

| 参数 | 含义 |
|-------|----------|
| conf | 模型配置的yaml文件路径 |
| input_dir | 需要预测的图片目录 |

**配置文件**的样例以及字段注释说明请参考: [conf/humanseg.yaml](inference/conf/humanseg.yaml)

样例程序会扫描input_dir目录下的所有图片，并生成对应的预测结果图片。

文件`14.jpg`预测的结果存储在`14_jpg.png`中，可视化结果在`14_jpg_scoremap.png`中， 原始尺寸的预测结果在`14_jpg_recover.png`中。

输入原图  
![avatar](inference/images/humanseg/demo.jpg)

输出预测结果   
![avatar](inference/images/humanseg/demo_jpg_recover.png)
