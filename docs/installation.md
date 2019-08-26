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
 
更多安装方式详情可以查看 [PaddlePaddle快速开始](https://www.paddlepaddle.org.cn/start)
 

## 2. 下载PaddleSeg代码
 
```
git clone https://github.com/PaddlePaddle/PaddleSeg
```
 

## 3. 安装PaddleSeg依赖
 
```
pip install -r requirements.txt
```
 

## 4. 本地流程测试
 
通过执行以下命令，会完整执行数据下载，训练，可视化，预测模型导出四个环节，用于验证PaddleSeg安装和依赖是否正常。
 
```
python test/local_test_cityscapes.py
```