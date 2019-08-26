# PaddleSegServing 
## 1.简介
PaddleSegServing是基于PaddleSeg开发的实时图像分割服务的企业级解决方案。用户仅需关注模型本身，无需理解模型模型的加载、预测以及GPU/CPU资源的并发调度等细节操作，通过设置不同的参数配置，即可根据自身的业务需求定制化不同图像分割服务。目前，PaddleSegServing支持人脸分割、城市道路分割、宠物外形分割模型。本文将通过一个人脸分割服务的搭建示例，展示PaddleSeg服务通用的搭建流程。

## 2.预编译版本安装及搭建服务流程
### 2.1. 下载预编译的PaddleSegServing
预编译版本在Centos7.6系统下编译，如果想快速体验PaddleSegServing，可在此系统下下载预编译版本进行安装。预编译版本有两个，一个是针对有GPU的机器，推荐安装GPU版本PaddleSegServing。另一个是CPU版本PaddleServing，针对无GPU的机器。

#### 2.1.1. 下载并解压GPU版本PaddleSegServing

```bash
cd ~
wget -c XXXX/PaddleSegServing.centos7.6_cuda9.2_gpu.tar.gz
tar xvfz PaddleSegServing.centos7.6_cuda9.2_gpu.tar.gz
```

#### 2.1.2. 下载并解压CPU版本PaddleSegServing

```bash
cd ~
wget -c XXXX/PaddleSegServing.centos7.6_cuda9.2_cpu.tar.gz
tar xvfz PaddleSegServing.centos7.6_cuda9.2_gpu.tar.gz
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

### 2.2. 运行PaddleSegServing
本节将介绍如何运行以及测试PaddleSegServing。

#### 2.2.1. 搭建人脸分割服务
搭建人脸分割服务只需完成一些配置文件的编写即可，其他分割服务的搭建流程类似。

##### 2.2.1.1. 下载人脸分割模型文件，并将其复制到相应目录。
```bash
# 下载人脸分割模型
wget -c https://paddleseg.bj.bcebos.com/inference_model/deeplabv3p_xception65_humanseg.tgz
tar xvfz deeplabv3p_xception65_humanseg.tgz
# 安装模型
cp -r deeplabv3p_xception65_humanseg seg-serving/bin/data/model/paddle/fluid
```


##### 2.2.1.2. 配置参数文件

参数文件如，PaddleSegServing仅新增一个配置文件seg_conf.yaml,用来指定具体分割模型的一些参数，如均值、方差、图像尺寸等。该配置文件可在gflags.conf中通过--seg_conf_file指定。

其他配置文件的字段解释可参考以下链接：https://github.com/PaddlePaddle/Serving/blob/develop/doc/SERVING_CONFIGURE.md  （TODO：介绍seg_conf.yaml中每个字段的含义）

```bash
conf/
├── gflags.conf
├── model_toolkit.prototxt
├── resource.prototxt
├── seg_conf.yaml
├── service.prototxt
└── workflow.prototxt
```

#### 2.2.2 运行服务端程序

```bash
# 1. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:$LD_LIBRARY_PATH
# 2. 切换到bin目录，运行服务端程序
cd ~/serving/build/output/demo/seg-serving/bin/
./seg-serving
```
#### 2.2.3.运行客户端程序进行测试 (建议在windows、mac测试，可直接查看分割后的图像)

客户端程序是用Python3编写的，代码简洁易懂，可以通过运行客户端验证服务的正确性以及性能表现。

```bash
# 使用Python3.6，需要安装opencv-python、requests、numpy包（建议安装anaconda）
cd tools
vim image_seg_client.py (修改IMAGE_SEG_URL变量，改成服务端的ip地址)
python3.6 image_seg_client.py
# 当前目录下可以看到生成出分割结果的图片。
```

## 3. 源码编译安装及搭建服务流程 (可选)
源码编译安装时间较长，一般推荐在centos7.6下安装预编译版本进行使用。如果您系统版本非centos7.6或者您想进行二次开发，请点击以下链接查看[源码编译安装流程](./COMPILE_GUIDE.md)。
