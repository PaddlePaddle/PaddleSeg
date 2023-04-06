# 安装说明
## 1 环境要求

- PaddlePaddle (版本不低于2.3)
- PaddleDetection
- PaddleSeg
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1
- cuDNN >= 7.6

## 2 本地安装说明

下载PaddleSeg, 进入./contrib/QualityInspector/目录
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd contrib/QualityInspector/
```

执行如下命令，完成PaddleDetection，PaddleSeg等其他第三方库的安装：
```
pip install -r requirements.txt
```
