# 安装PaddlePaddle

版本要求
* PaddlePaddle >= 1.6.1
* Python 2.7 or 3.5+

更多详细安装信息如CUDA版本、cuDNN版本等兼容信息请查看[PaddlePaddle官网安装](https://www.paddlepaddle.org.cn/install/doc/index)

### pip安装
 
由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg.
 
```
pip install paddlepaddle-gpu
```

### Conda安装
 
PaddlePaddle最新版本1.5支持Conda安装，可以减少相关依赖安装成本，conda相关使用说明可以参考[Anaconda](https://www.anaconda.com/distribution/)
 
```
conda install -c paddle paddlepaddle-gpu cudatoolkit=9.0
```
 
 * 如果有多卡训练需求，请安装 NVIDIA NCCL >= 2.4.7，并在Linux环境下运行
 
更多安装方式详情可以查看 [PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)
 
