简体中文|[English](install.md)
# 安装文档


## 环境要求

- PaddlePaddle (版本不低于2.2)
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1
- cuDNN >= 7.6

## 一、本地安装说明

### 1. 安装PaddlePaddle

请参考[快速安装文档](https://www.paddlepaddle.org.cn/install/quick)或者[详细安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)，安装PaddlePaddle （要求不低于2.2版本，推荐安装最新版本）。

比如Linux、CUDA 10.1，使用pip安装GPU版本，执行如下命令。

```
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

使用如下命令验证PaddlePaddle是否安装成功，并且查看版本。

```
# 在Python解释器中顺利执行如下命令
>>> import paddle
>>> paddle.utils.run_check()

# 如果命令行出现以下提示，说明PaddlePaddle安装成功
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# 查看PaddlePaddle版本
>>> print(paddle.__version__)

```

### 2. 下载PaddleSeg代码

从Github下载PaddleSeg代码。

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

如果连不上Github，可以从Gitee下载PaddleSeg代码，但是Gitee上代码可能不是最新。

```
git clone https://gitee.com/paddlepaddle/PaddleSeg.git
```

### 3. 安装PaddleSeg依赖

```
cd PaddleSeg
pip install -r requirements.txt
```

如果安装时出现版本错误，可以尝试删除旧版本，重新运行该脚本。

### 4.确认环境安装成功

在PaddleSeg目录下执行如下命令，会进行简单的单卡训练和单卡预测。

```
sh tests/run_check_install.sh
```

查看执行输出的log，没有报错，则验证安装成功。

然后，基于PaddleSeg，你可以使用配置文件的方式来训练、验证、测试和导出模型。


### 5. 安装PaddleSeg包（可选）

PaddleSeg也支持API方式灵活开发。 API开发方式要求大家将PaddleSeg安装到Python库中。


执行如下命令，从源码编译安装PaddleSeg包。
```
cd PaddleSeg
python setup.py install
```

或者直接安装发布的PaddleSeg包。
```
pip install paddleseg
```

## 二、使用Docker快速体验PaddleSeg

Docker是一种开源工具，用于在和系统本身环境相隔离的环境中构建、发布和运行各类应用程序。如果您没有Docker运行环境，请参考[Docker 官网](https://www.docker.com/)进行安装，如果您准备使用GPU版本镜像，还需要提前安装好[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

我们提供了包含最新PaddleSeg代码的docker镜像，并预先安装好了所有的环境和库依赖，您只需要**拉取并运行docker镜像**，无需其他任何额外操作，即可开始享用PaddleSeg的所有功能。

在[Docker Hub](https://hub.docker.com/repository/docker/paddlecloud/paddleseg)中获取这些镜像及相应的使用指南，包括CPU、GPU、ROCm 版本。

如果您对自动化制作docker镜像感兴趣，或有自定义需求，请访问[PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton)做进一步了解。
