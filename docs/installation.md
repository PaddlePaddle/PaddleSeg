# PaddleSeg 安装说明

## 推荐开发环境

* Python2.7 or 3.5+
* CUDA 9.2
* cudnn v7.1



## 1. 安装PaddlePaddle

### pip安装
 
由于图像分割任务模型计算量大，强烈推荐在GPU版本的paddlepaddle下使用PaddleSeg.
 
```
pip install paddlepaddle-gpu
```

### Conda安装
 
PaddlePaddle最新版本1.5支持Conda安装，可以减少相关依赖安装成本，conda相关使用说明可以参考[Anaconda](https://www.anaconda.com/distribution/)
 
```
conda install -c paddle paddlepaddle-gpu cudatoolkit=9.0
```
 
更多安装方式详情可以查看 [PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)
 

## 2. 下载PaddleSeg代码
 
```
git clone https://github.com/PaddlePaddle/PaddleSeg
```
 

## 3. 安装PaddleSeg依赖
 
```
cd PaddleSeg
pip install -r requirements.txt
```
