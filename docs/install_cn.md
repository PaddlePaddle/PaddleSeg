简体中文|[English](install.md)
# 安装文档


## 环境要求

- PaddlePaddle 2.1 (获取底层API支持)
- OS 64位操作系统 （运行64位程序）
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本 （提供环境支持）
- CUDA >= 10.1 （NVIDIA GPU 并行计算框架）
- cuDNN >= 7.6 （NVIDIA GPU 加速库）

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.1.0.post101 -i https://paddlepaddle.org.cn/whl/mkl/stable.html

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"

# 如果命令行出现以下提示，说明PaddlePaddle安装成功
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```



## 2.下载PaddleSeg代码
```
git clone https://github.com/PaddlePaddle/PaddleSeg
```
## 3.安装PaddleSeg依赖
```
cd PaddleSeg
pip install -r requirements.txt

#如果安装时出现版本错误，可以尝试删除旧版本，重新运行该脚本。
```
## 4.确认环境安装成功

执行下面命令，并在PaddleSeg/output文件夹中出现预测结果，则证明安装成功

```python
python predict.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path https://bj.bcebos.com/paddleseg/dygraph/optic_disc/bisenet_optic_disc_512x512_1k/model.pdparams\
       --image_path docs/images/optic_test_image.jpg \
       --save_dir output/result
```
