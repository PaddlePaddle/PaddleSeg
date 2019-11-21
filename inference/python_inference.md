# PaddleSeg Python 预测部署方案

本说明文档旨在提供一个跨平台的图像分割模型的Python预测部署方案，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成图像分割的任务。

## 前置条件

* Python2.7+，Python3+
* pip，pip3

## 主要目录和文件

```
inference
├── infer.py # 完成预测、可视化的Python脚本
└── requirements.txt # 预测部署脚本所依赖的库
```

### Step1:安装PaddlePaddle

可参考以下链接，选择合适版本的PaddlePaddle进行安装。[PaddlePaddle安装教程](https://www.paddlepaddle.org.cn/install/doc/)

### Step2:安装Python依赖包

在inference目录下，安装相应的Python预测依赖包
```bash
pip install -r requirements.txt
```
因为预测部署中需要使用opencv，所以还需要安装相关的动态链接库。相关操作如下：

Ubuntu 下安装相关链接库：
```bash
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
```

CentOS 下安装相关链接库：
```bash
yum install -y libXext libSM libXrender
```
### Step3:预测

在终端输入以下命令进行预测。
```
python infer.py --conf=/path/to/XXX.yaml --input_dir/path/to/images_directory --use_pr=False
```

预测使用的三个命令参数说明如下：

| 参数 | 含义 |
|-------|----------|
| conf | 模型配置的Yaml文件路径 |
| input_dir | 需要预测的图片目录 |
| use_pr | 是否使用优化模型，默认为False|

* 优化模型：对于图像分割模型，由于模型输入的数据需要使用CPU对读取的图像数据进行预处理，预处理时长较长，为了降低在使用GPU进行端到端预测时的延时，优化模型把预处理部分融入到模型当中。在使用GPU进行预测时，优化模型的预处理部分将会在GPU上进行，大大降低了端到端延时。可使用新版的模型导出工具导出优化模型。

![avatar](images/humanseg/demo2.jpeg)

输出预测结果   
![avatar](images/humanseg/demo2.jpeg_result.png)