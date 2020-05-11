# 人像分割Python预测部署方案

本方案基于Python实现，最小化依赖并把所有模型加载、数据预处理、预测、光流处理等后处理都封装在文件`infer.py`中，用户可以直接使用或集成到自己项目中。


## 前置依赖
- Windows(7,8,10) / Linux (Ubuntu 16.04) or MacOS 10.1+
- Paddle 1.7+
- Python 3.6+

注意：
1. 仅支持Paddle 1.7以上版本
2. MacOS上不支持GPU预测

其它未涉及情形，能正常安装`Paddle` 和`OpenCV`通常都能正常使用。


## 安装依赖

执行如下命令

```shell
pip install -r requirements.txt
```

## 运行


1. 输入图片进行分割
```
python infer.py --model_dir /PATH/TO/INFERENCE/MODEL --img_path /PATH/TO/INPUT/IMAGE
```

预测结果会保存为`result.jpeg`。
2. 输入视频进行分割
```shell
python infer.py --model_dir /PATH/TO/INFERENCE/MODEL --video_path /PATH/TO/INPUT/VIDEO
```

预测结果会保存在`result.avi`。

3. 使用摄像头视频流
```shell
python infer.py --model_dir /PATH/TO/INFERENCE/MODEL --use_camera True
```
预测结果会通过可视化窗口实时显示。

**注意：**


`GPU`默认关闭, 如果要使用`GPU`进行加速，则先运行
```
export CUDA_VISIBLE_DEVICES=0
```
然后在前面的预测命令中增加参数`--use_gpu True`即可。
