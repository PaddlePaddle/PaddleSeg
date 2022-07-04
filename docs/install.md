English | [简体中文](install_cn.md)


## Environment Requirements

- PaddlePaddle (the version >= 2.2)
- OS: 64-bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64-bit version
- pip/pip3(9.0.1+)，64-bit version
- CUDA >= 10.1
- cuDNN >= 7.6

## Installation

### 1. Install PaddlePaddle

Please refer to the [installation doc](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html) to install PaddlePaddle (the version >= 2.2).

Highly recommend you install the GPU version of PaddlePaddle, due to the large overhead of segmentation models, otherwise, it could be out of memory while running the models.

For example, run the following command to install Paddle with pip for Linux, CUDA 10.1.

```
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```


In Python interpreter, run the following command to confirm whether PaddlePaddle is installed successfully

```
>>> import paddle
>>> paddle.utils.run_check()

# If the following prompt appears on the command line, the PaddlePaddle installation is successful.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# Confirm PaddlePaddle version
>>> print(paddle.__version__)

```

### 2. Download Repo

Clone the PaddleSeg repo from Github.

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

### 3. Install Dependencies

```
cd PaddleSeg
pip install -r requirements.txt
```

If a version error occurs during installation, you can try to delete the old version and re-run the script.
### 4. Verify Installation

In the root of PaddleSeg, run the following command. If there are no error in terminal log, you can use PaddleSeg to train, validate, test and export models with config method.

```
cd PaddleSeg
sh tests/run_check_install.sh
```

### 5. Install PaddleSeg (Optional)

Besides, PaddleSeg also supports api method for flexible development, which requires the PaddleSeg is installed in python libs.

Run the following command to compile and install PaddleSeg from source code.

```
cd PaddleSeg
python setup.py install
```

Or, you can install the release PaddleSeg.
```
pip install paddleseg
```

## Use PaddleSeg with Docker

Docker is an open-source tool to build, ship, and run distributed applications in an isolated environment. If you  do not have a Docker environment, please refer to [Docker](https://www.docker.com/). If you will use GPU version, you also need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

We provide docker images containing the latest PaddleSeg code, and all environment and package dependencies are pre-installed. All you have to do is to **pull and run the docker image**. Then you can enjoy PaddleSeg without any extra steps.

Get these images and guidance in [docker hub](https://hub.docker.com/repository/docker/paddlecloud/paddleseg), including CPU, GPU, ROCm environment versions.

If you have some customized requirements about automatic building docker images, you can get it in github repo [PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton).
