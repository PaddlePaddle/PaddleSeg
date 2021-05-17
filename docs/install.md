# PaddleSeg安装步骤

## 1.安装PaddlePaddle

版本要求

* PaddlePaddle>=1.6.1

* Python 2.7 or 3.5+

### pip 安装
由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.

```
pip install paddlepaddle-gpu
```
### Conda 安装
PaddlePaddle最新版本1.5支持Conda安装，可以减少相关依赖安装成本，conda相关使用说明可以参考[Anaconda](https://www.anaconda.com/distribution/)

* 如果有多卡训练需求，请安装 NVIDIA NCCL >=2.4.7，并在LINUX环境下运行
更多安装方式详情可以查看 [PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)

## 2.下载PaddleSeg代码
```
git clone https://github.com/PaddlePaddle/PaddleSeg
```
## 3.安装PaddleSeg依赖
```
cd PaddleSeg
pip install -r requirements.txt
```