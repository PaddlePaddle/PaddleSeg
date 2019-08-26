# 源码编译安装及搭建服务流程 
本文将介绍源码编译安装以及在服务搭建流程。

## 1. 系统依赖项

依赖项 | 验证过的版本
   -- | --
Linux | Centos 6.10 / 7
CMake | 3.0+
GCC   | 4.8.2/5.4.0
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


## 2. 安装依赖项

以下流程在百度云CentOS7.5+CUDA9.2环境下进行。
### 2.1. 安装openssl、Go编译器以及bzip2

```bash
yum -y install openssl openssl-devel golang bzip2-libs bzip2-devel
```

### 2.2. 安装GPU预测的依赖项（如果需要使用GPU预测，必须执行此步骤）
#### 2.2.1. 安装配置CUDA9.2以及cuDNN 7.1.4
该百度云机器已经安装CUDA以及cuDNN，仅需复制相关头文件与链接库 

```bash
# 看情况确定是否需要安装 cudnn
# 进入 cudnn 根目录
cd /home/work/cudnn/cudnn7.1.4
# 拷贝头文件
cp include/cudnn.h /usr/local/cuda/include/
# 拷贝链接库
cp lib64/libcudnn* /usr/local/cuda/lib64/
# 修改头文件、链接库访问权限
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

#### 2.2.2. 安装nccl库

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

### 2.3. 安装 cmake 3.15 
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

### 2.4. 为依赖库增加相应的软连接

  现在Linux系统中大部分链接库的名称都以版本号作为后缀，如libcurl.so.4.3.0。这种命名方式最大的问题是，CMakeList.txt中find_library命令是无法识别使用这种命名方式的链接库，会导致CMake时候出错。由于本项目是用CMake构建，所以务必保证相应的链接库以 .so 或 .a为后缀命名。解决这个问题最简单的方式就是用创建一个软连接指向相应的链接库。在百度云的机器中，只有curl库的命名方式有问题。所以命令如下：（如果是其他库，解决方法也类似）：
  
```bash
ln -s /usr/lib64/libcurl.so.4.3.0 /usr/lib64/libcurl.so
```


### 2.5. 编译安装PaddleServing 
下列步骤介绍CPU版本以及GPU版本的PaddleServing编译安装过程。

```bash
# Step 1. 在～目录下下载paddle-serving代码
cd ~
git clone https://github.com/PaddlePaddle/serving.git
# Step 2. 进入serving目录，创建build目录编译、安装
cd serving
mkdir build 
cd build
# Step 3. 以下为生成GPU版本的makefile，生成CPU版本的makefile执行 cmake -DWITH_GPU=OFF ..
cmake -DWITH_GPU=ON -DCUDNN_ROOT=/usr/local/cuda/lib64 ..
# Step 4. nproc 可以输出当前机器的核心数,利用多核进行编译。如果make时候报错退出，可以多执行几次make解决
make -j$(nproc)
# Step 5. 安装
make install
# Step 6. 安装后可以看PaddleServing的目录结构如下
serving
├── build
├── cmake
├── CMakeLists.txt
├── configure
├── CONTRIBUTING.md
├── cube
├── demo-client
├── demo-serving 
│   ├── CMakeLists.txt
│   ├── conf        # demo-serving 的配置文件目录
│   ├── data        # 模型文件以及参数文件的目录
│   ├── op          # 数据处理的源文件目录
│   ├── proto       # 数据传输的proto文件目录
│   └── scripts 
├── doc
├── inferencer-fluid-cpu
├── inferencer-fluid-gpu
├── kvdb
├── LICENSE
├── pdcodegen
├── predictor
├── README.md
├── sdk-cpp
└── tools
```

### 2.6. 安装PaddleSegServing

```bash
# Step 1. 在～目录下下载PaddleSeg代码
git clone http://gitlab.baidu.com/Paddle/PaddleSeg.git
# Step 2. 进入PaddleSeg的serving目录（注意区分PaddleServing的serving目录），并将seg-serving目录复制到PaddleServing的serving目录下
cd PaddleSeg/serving
cp -r seg-serving ~/serving
# 复制后PaddleServing的目录结构如下
serving
├── build
├── cmake
├── CMakeLists.txt
├── configure
├── CONTRIBUTING.md
├── cube
├── demo-client
├── demo-serving 
├── doc
├── inferencer-fluid-cpu
├── inferencer-fluid-gpu
├── kvdb
├── LICENSE
├── pdcodegen
├── predictor
├── README.md
├── sdk-cpp
├── seg-serving   # 此为新增的目录
└── tools

# Step 3. 修改PaddleServing的serving目录下的CMakeLists.txt
cd ~/serving
vim CMakeLists.txt
# Step 4. 倒数第二行加入代码，使得seg-serving下的代码可与PaddleServing一起编译
add_subdirectory(seg-serving)
# Step 5. 进入PaddleServing的build目录，编译安装PaddleSegServing
cd ~/serving/build
make -j$(nproc)
make install
# Step 6. 完成安装后,可以看到执行文件的目录结构如下
build
├── boost_dummy.c
├── CMakeCache.txt
├── CMakeFiles
├── cmake_install.cmake
├── configure
├── demo-client
├── error
├── human-seg-serving
├── inferencer-fluid-cpu
├── inferencer-fluid-gpu
├── info
├── install_manifest.txt
├── kvdb
├── libboost.a
├── log
├── Makefile
├── output          # 所有服务端的执行文件、配置文件、数据文件均安装到此目录下
│   ├── bin
│   ├── demo
│   │   ├── client
│   │   ├── db_func
│   │   ├── db_thread
│   │   ├── seg-serving  
│   │   │   └── bin
│   │   │       ├── conf    # 配置文件目录
│   │   │       ├── data    # 数据模型文件、参数文件目录
│   │   │       ├── seg-serving #可执行文件
│   │   │       ├── kvdb
│   │   │       ├── libiomp5.so
│   │   │       ├── libmklml_gnu.so
│   │   │       ├── libmklml_intel.so
│   │   │       └── log
│   │   ├── kvdb_test
│   │   └── serving
│   ├── include
│   └── lib
├── Paddle
├── pdcodegen
├── predictor
├── sdk-cpp
├── seg-serving
└── third_party
```

## 3. 运行PaddleSegServing

### 3.1. 搭建人脸分割服务
搭建人脸分割服务只需完成一些配置文件的编写即可。与预编译版本的搭建大致相同，但模型文件、参数文件放置的目录略有不同。

#### 3.1.1. 下载人脸分割模型文件，并将其复制到PaddleSegServing相应目录。
可参考[预编译安装流程](./README.md)中2.2.1.1节。模型文件放置的目录在
～/serving/seg-serving/data/model/paddle/fluid/。


#### 3.1.2. 配置参数文件。
可参考[预编译安装流程](./README.md)中2.2.1.2节。配置文件的目录在～/serving/seg-serving/conf。

### 3.2 安装模型文件、配置文件。

```bash
cd ~/serving/build
make install
```

### 3.3 运行服务端程序
可参考[预编译安装流程](./README.md)中2.2.2节。可执行文件在该目录下：～/serving/build/output/demo/seg-serving/bin/。

### 3.4 运行客户端程序进行测试。
可参考[预编译安装流程](./README.md)中2.2.3节。
