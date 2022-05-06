# 在Linux上测试分割模型速度

本文档介绍使用Paddle Inference的C++接口在Linux服务器端(NV GPU)测试分割模型速度。

## 环境准备

### 准备基础环境

在Nvidia GPU上测试模型，必须安装必CUDA、cudnn。此外，需要下载TensorRT库文件。

此处提供两个版本的CUDA、cudnn、TensorRT文件下载。

```
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.1-cudnn7.6-trt6.0.tar
wget https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/cuda10.2-cudnn8.0-trt7.1.tgz
```

下载解压后，CUDA和cudnn可以参考网上文档或者官方文档([Cuda Doc](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [cudnn Doc](https://docs.nvidia.com/deeplearning/cudnn/install-guide/))进行安装。TensorRT只需要设置库路径，比如：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/TensorRT-7.1.3.4/lib
```

### 准备Paddle Inference C++预测库

在Nvidia GPU上测试模型，进入[C++预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载对应CUDA、Cudnn、TRT、GCC版本的PaddleInference库。

> 不同C++预测库可以根据名字进行区分。请根据机器的操作系统、CUDA版本、cudnn版本、使用MKLDNN或者OpenBlas、是否使用TenorRT等信息，选择准确版本。（建议选择版本>=2.3rc的预测库）

下载`paddle_inference.tgz`压缩文件，解压到本示例的根目录下。

### 2.3 安装其他库

本示例使用OpenCV读取图片，所以需要安装OpenCV。
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

本示例使用Gflags和Glog，执行如下命令安装。
```
git clone https://github.com/gflags/gflags.git
mkdir -p gflags/build
cd gflags/build
cmake ..
make -j
make install
```

```
git clone https://github.com/google/glog
mkdir -p glog/build
cd glog/build
cmake ..
make -j
make install
```

## 准备模型

将多个预测模型保存到一个文件夹下，进行批量测试，比如：

```
infer_models
├── dwcoa_pp_liteseg_stdc1_512x512
├── dwcoa_pp_liteseg_stdc2_512x512
├── dwcoa_ocrnet_hrnet18_512x512
```

## 执行测试

打开`run_seg_speed.sh`文件，根据实际情况，设置如下变量，其他变量也可以自行修改。
* paddle_root为PaddleInference库路径
* tensorrt_root为TensorRT库路径
* 设置model_dir为保存预测模型的路径
* target_width和target_height为输入图像resize后的宽高
* save_path为结果信息的保存文件

`run_seg_speed.sh`文件，默认参数是使用GPU、开启TRT、使用Auto tune收集shape，然后进行预测。

执行`sh run_seg_speed.sh`。

执行结束后，查看保存结果信息的文件，其中包括测试的配置信息、模型名字、预处理时间、执行时间。
