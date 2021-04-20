# 源码编译安装及搭建服务流程
本文将介绍源码编译安装以及在服务搭建流程。编译前确保PaddleServing的依赖项安装完毕。依赖安装教程请前往[PaddleSegServing 依赖安装](./README.md).

## 1. 编译安装PaddleServing
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

## 2. 安装PaddleSegServing

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
