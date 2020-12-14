# PaddlePaddle 集成TensorRT 编译安装文档

本文以`Ubuntu 16.04` 为例说明如何编译支持`TensorRT`的`PaddlePaddle`包。

## 1. 确认依赖的基础软件环境
- Python 2.7+ / Python 3.5+
- CUDA 9.0
- CuDNN 7.5
- cmake 3.10
- gcc 4.8.3

## 2. 安装 TensorRT 5.1

请参考Nvidia的[官方安装教程](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)

## 3. 编译 PaddlePaddle

 这里假设`Python`版本为`3.7`以及`CUDA` `cuDNN` `TensorRT`安装路径如下：
```bash
# 假设 cuda 安装路径
/usr/local/cuda-9.0/
# 假设 cudnn 安装路径
/usr/local/cudnn-7.5/
# 假设 tensorRT 安装路径
/usr/local/TensorRT-5.1/
```

那么执行如下命令进行编译(请根据实际情况修改)：

```bash
# 下载 Paddle 代码
git clone https://github.com/PaddlePaddle/Paddle.git
# 进入 Paddle 目录
cd Paddle
# 创建编译目录
mkdir build
cd build
# cmake 编译
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDNN_ROOT=/usr/local/cudnn-7.5/ \
      -DCMAKE_INSTALL_PREFIX=`pwd`/output \
      -DWITH_PYTHON=ON   \
      -DON_INFER=ON \
      -DWITH_GPU=ON \
      -DCUDA_ARCH_NAME=Auto \
      -DTENSORRT_INCLUDE_DIR=/usr/local/TensorRT-5.1.5.0/include \
      -DTENSORRT_LIBRARY=/usr/local/TensorRT-5.1.5.0/lib \
      -DPY_VERSION=3.7 \

make -j20
make install

```

编译完成后，在`build/python/dist`目录下会生成一个`whl`包，执行下面的命令安装即可：
```bash
pip install -U xxxx.whl
```

## 4. 验证安装
进入 `python`, 执行以下代码：
```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```
如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。
