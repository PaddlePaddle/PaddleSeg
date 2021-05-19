# Ubuntu系统下依赖项的安装教程
运行PaddleSegServing需要系统安装一些依赖库。在不同发行版本的Linux系统下，安装依赖项的具体命令略有不同，以下介绍在Ubuntu 16.07下安装依赖项的方法。

## 1. 安装ssl、go、python、bzip2、crypto.

```bash
sudo apt-get install golang-1.10 python2.7 libssl1.0.0 libssl-dev libssl-doc libcrypto++-dev libcrypto++-doc libcrypto++-utils libbz2-1.0 libbz2-dev
```

## 2. 为ssl、crypto、curl链接库添加软连接

```bash
ln -s /lib/x86_64-linux-gnu/libssl.so.1.0.0 /usr/lib/x86_64-linux-gnu/libssl.so
ln -s /lib/x86_64-linux-gnu/libcrypto.so.1.0.0 /usr/lib/x86_64-linux-gnu/libcrypto.so.10
ln -s /usr/lib/x86_64-linux-gnu/libcurl.so.4.4.0 /usr/lib/x86_64-linux-gnu/libcurl.so
```

## 3. 安装GPU依赖项（如果需要使用GPU预测，必须执行此步骤）
### 3.1. 安装配置CUDA 9.2以及cuDNN 7.1.4
方法与[预编译安装流程](README.md) 2.2.2.1节一样。

### 3.2. 安装nccl库（如果已安装nccl 2.4.7请忽略该步骤）

```bash
# 下载nccl相关的deb包
wget -c --no-check-certificate https://paddleseg.bj.bcebos.com/serving/nccl-repo-ubuntu1604-2.4.8-ga-cuda9.2_1-1_amd64.deb
sudo apt-key add /var/nccl-repo-2.4.8-ga-cuda9.2/7fa2af80.pub
# 安装deb包
sudo dpkg -i nccl-repo-ubuntu1604-2.4.8-ga-cuda9.2_1-1_amd64.deb
# 更新索引
sudo apt update
# 安装nccl库
sudo apt-get install libnccl2 libnccl-dev
```

## 4. 安装cmake 3.15
如果机器没有安装cmake或者已安装cmake的版本低于3.0，请执行以下步骤

```bash
# 如果原来的已经安装低于3.0版本的cmake，请先卸载原有低版本 cmake
sudo apt-get autoremove cmake
```
其余安装cmake的流程请参考以下链接[预编译安装流程](README.md) 2.2.3节。

## 5. 安装PaddleSegServing
### 5.1. 下载并解压GPU版本PaddleSegServing

```bash
cd ~
wget -c --no-check-certificate https://paddleseg.bj.bcebos.com/serving/paddle_seg_serving_ubuntu16.07_gpu_cuda9.2.tar.gz
tar xvfz PaddleSegServing.ubuntu16.07_cuda9.2_gpu.tar.gz seg-serving
```

### 5.2. 下载并解压CPU版本PaddleSegServing

```bash
cd ~
wget -c --no-check-certificate https://paddleseg.bj.bcebos.com/serving%2Fpaddle_seg_serving_ubuntu16.07_cpu.tar.gz
tar xvfz PaddleSegServing.ubuntu16.07_cuda9.2_gpu.tar.gz seg-serving
```

## 6. gcc版本问题
在Ubuntu 16.07系统中，默认的gcc版本为5.4.0。而目前PaddleSegServing仅支持gcc 4.8编译，所以如果测试的机器gcc版本为5.4，请先进行降级（无需卸载原有的gcc）。

```bash
# 安装gcc 4.8
sudo apt-get install gcc-4.8
# 查看是否成功安装gcc4.8
ls /usr/bin/gcc*
# 设置gcc4.8的优先级，使其能被gcc命令优先连接gcc4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
# 查看设置结果（非必须）
sudo update-alternatives --config gcc
```
