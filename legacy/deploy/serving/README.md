# PaddleSegServing
## 1.简介
PaddleSegServing是基于PaddleSeg开发的实时图像分割服务的企业级解决方案。用户仅需关注模型本身，无需理解模型模型的加载、预测以及GPU/CPU资源的并发调度等细节操作，通过设置不同的参数配置，即可根据自身的业务需求定制化不同图像分割服务。目前，PaddleSegServing支持人脸分割、城市道路分割、宠物外形分割模型。本文将通过一个人脸分割服务的搭建示例，展示PaddleSeg服务通用的搭建流程。

## 2.预编译版本安装及搭建服务流程
运行PaddleSegServing需要依赖其他的链接库，请保证在下载安装前系统环境已经具有相应的依赖项。
安装以及搭建服务的流程均在Centos和Ubuntu系统上验证。以下是Centos系统上的搭建流程，Ubuntu版本的依赖项安装流程介绍在[Ubuntu系统下依赖项的安装教程](UBUNTU.md)。

### 2.1. 系统依赖项
依赖项 | 验证过的版本
   -- | --
Linux | Centos 6.10 / 7, Ubuntu16.07
CMake | 3.0+
GCC   | 4.8.2
Python| 2.7
GO编译器| 1.9.2
openssl| 1.0.1+
bzip2  | 1.0.6+

如果需要使用GPU预测，还需安装以下几个依赖库

 GPU库   | 验证过的版本
   -- | --
CUDA  | 9.2
cuDNN | 7.1.4
nccl  | 2.4.7

### 2.2. 安装依赖项

#### 2.2.1. 安装openssl、Go编译器以及bzip2

```bash
yum -y install openssl openssl-devel golang bzip2-libs bzip2-devel
```
#### 2.2.2. 安装GPU预测的依赖项（如果需要使用GPU预测，必须执行此步骤）
#### 2.2.2.1. 安装配置CUDA 9.2以及cuDNN 7.1.4
请确保正确安装CUDA 9.2以及cuDNN 7.1.4. 以下为安装CUDA和cuDNN的官方教程。

```bash
安装CUDA教程: https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmnetwork

安装cuDNN教程: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
```

#### 2.2.2.2. 安装nccl库（如果已安装nccl 2.4.7请忽略该步骤）

```bash
# 下载文件 nccl-repo-rhel7-2.4.7-ga-cuda9.2-1-1.x86_64.rpm
wget -c https://paddlehub.bj.bcebos.com/serving/nccl-repo-rhel7-2.4.7-ga-cuda9.2-1-1.x86_64.rpm
# 安装nccl的repo
rpm -i nccl-repo-rhel7-2.4.7-ga-cuda9.2-1-1.x86_64.rpm
# 更新索引
yum -y update
# 安装包
yum -y install libnccl-2.4.7-1+cuda9.2 libnccl-devel-2.4.7-1+cuda9.2 libnccl-static-2.4.7-1+cuda9.2
```

### 2.2.3. 安装 cmake 3.15
如果机器没有安装cmake或者已安装cmake的版本低于3.0，请执行以下步骤

```bash
# 如果原来的已经安装低于3.0版本的cmake，请先卸载原有低版本 cmake
yum -y remove cmake
# 下载源代码并解压
wget -c https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0.tar.gz
tar xvfz cmake-3.15.0.tar.gz
# 编译cmake
cd cmake-3.15.0
./configure
make -j4
# 安装并检查cmake版本
make install
cmake --version
# 在cmake-3.15.0目录中，将相应的头文件目录（curl目录，为PaddleServing的依赖头文件目录）拷贝到系统include目录下
cp -r Utilities/cmcurl/include/curl/ /usr/include/
```

### 2.2.4. 为依赖库增加相应的软连接

  现在Linux系统中大部分链接库的名称都以版本号作为后缀，如libcurl.so.4.3.0。这种命名方式最大的问题是，CMakeList.txt中find_library命令是无法识别使用这种命名方式的链接库，会导致CMake时候出错。由于本项目是用CMake构建，所以务必保证相应的链接库以 .so 或 .a为后缀命名。解决这个问题最简单的方式就是用创建一个软连接指向相应的链接库。在百度云的机器中，只有curl库的命名方式有问题。所以命令如下：（如果是其他库，解决方法也类似）：

```bash
ln -s /usr/lib64/libcurl.so.4.3.0 /usr/lib64/libcurl.so
```

### 2.3. 下载预编译的PaddleSegServing
预编译版本在Centos7.6系统下编译，如果想快速体验PaddleSegServing，可在此系统下下载预编译版本进行安装。预编译版本有两个，一个是针对有GPU的机器，推荐安装GPU版本PaddleSegServing。另一个是CPU版本PaddleServing，针对无GPU的机器。

#### 2.3.1. 下载并解压GPU版本PaddleSegServing

```bash
cd ~
wget -c --no-check-certificate https://paddleseg.bj.bcebos.com/serving/paddle_seg_serving_centos7.6_gpu_cuda9.2.tar.gz
tar xvfz PaddleSegServing.centos7.6_cuda9.2_gpu.tar.gz seg-serving
```

#### 2.3.2. 下载并解压CPU版本PaddleSegServing

```bash
cd ~
wget -c --no-check-certificate https://paddleseg.bj.bcebos.com/serving/paddle_seg_serving_centos7.6_cpu.tar.gz
tar xvfz PaddleSegServing.centos7.6_cuda9.2_gpu.tar.gz seg-serving
```

解压后的PaddleSegServing目录如下。
```bash
├── seg-serving  
    └── bin
        ├── conf    # 配置文件目录
        ├── data    # 数据模型文件、参数文件目录
        ├── seg-serving #可执行文件
        ├── kvdb
        ├── libiomp5.so
        ├── libmklml_gnu.so
        ├── libmklml_intel.so
        └── log
```

### 2.4 安装动态库
把 libiomp5.so, libmklml_gnu.so, libmklml_intel.so拷贝到/usr/lib。

```bash
cd seg-serving/bin/
cp libiomp5.so libmklml_gnu.so libmklml_intel.so /usr/lib
```


### 2.5. 运行PaddleSegServing
本节将介绍如何运行以及测试PaddleSegServing。

#### 2.5.1. 搭建人脸分割服务
搭建人脸分割服务只需完成一些配置文件的编写即可，其他分割服务的搭建流程类似。

#### 2.5.1.1. 下载人脸分割模型文件，并将其复制到相应目录。
```bash
# 下载人脸分割模型
wget -c https://paddleseg.bj.bcebos.com/inference_model/deeplabv3p_xception65_humanseg.tgz
tar xvfz deeplabv3p_xception65_humanseg.tgz
# 安装模型
cp -r deeplabv3p_xception65_humanseg seg-serving/bin/data/model/paddle/fluid
```


#### 2.5.1.2. 配置参数文件。参数文件如下。PaddleSegServing仅新增一个配置文件seg_conf.yaml,用来指定具体分割模型的一些参数，如均值、方差、图像尺寸等。该配置文件可在gflags.conf中通过--seg_conf_file指定。其他配置文件的字段解释可参考以下链接：https://github.com/PaddlePaddle/Serving/blob/develop/doc/SERVING_CONFIGURE.md

```bash
conf/
├── gflags.conf
├── model_toolkit.prototxt
├── resource.prototxt
├── seg_conf.yaml
├── service.prototxt
└── workflow.prototxt
```

以下为seg_conf.yaml文件内容以及每一个配置项的内容。

```bash
%YAML:1.0
# 输入到模型的图像的尺寸。会将任意图片resize到513*513尺寸的图像，再放入模型进行推测。
SIZE: [513, 513]
# 均值
MEAN: [104.008, 116.669, 122.675]
# 方差
STD: [1.0, 1.0, 1.0]
# 通道数
CHANNELS: 3
# 类别数量
CLASS_NUM: 2
# 加载的模型的名称，需要与model_toolkit.prototxt中对应模型的名称保持一致。
MODEL_NAME: "human_segmentation"
```

#### 2.5.2 运行服务端程序

```bash
# 1. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:$LD_LIBRARY_PATH
# 2. 切换到bin目录，运行服务端程序
cd ~/serving/build/output/demo/seg-serving/bin/
./seg-serving
```
#### 2.5.3.运行客户端程序
以下为PaddleSeg的目录结构，客户端在PaddleSeg/serving/tools目录。

```bash
PaddleSeg
├── configs
├── contrib
├── dataset
├── docs
├── inference
├── pdseg
├── README.md
├── requirements.txt
├── scripts
├── serving
│   ├── COMPILE_GUIDE.md
│   ├── imgs
│   ├── README.md
│   ├── requirements.txt        # 客户端程序依赖的包
│   ├── seg-serving
│   ├── tools                   # 客户端目录
│   │   ├── images              # 测试的图像目录，可放置jpg格式或其他三通道格式的图像，以jpg或jpeg作为文件后缀名
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   └── 3.jpg
│   │   └── image_seg_client.py # 客户端测试代码
│   └── UBUNTU.md
├── test
└── test.md
```
客户端程序使用Python3编写，通过下载requirements.txt中的python依赖包(`pip3 install -r requirements.txt`），用户可以在Windows、Mac、Linux等平台上正常运行该客户端，测试的图像放在PaddleSeg/serving/tools/images目录，用户可以根据自己需要把其他三通道格式的图片放置到该目录下进行测试。从服务端返回的结果图像保存在PaddleSeg/serving/tools目录下。

```bash
cd tools
vim image_seg_client.py (修改IMAGE_SEG_URL变量，改成服务端的ip地址)
python3.6 image_seg_client.py
# 当前目录可以看到生成出分割结果的图片。
```

## 3. 源码编译安装及搭建服务流程 (可选)
源码编译安装时间较长，一般推荐在centos7.6下安装预编译版本进行使用。如果您系统版本非centos7.6或者您想进行二次开发，请点击以下链接查看[源码编译安装流程](./COMPILE_GUIDE.md)。
