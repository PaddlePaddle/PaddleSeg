# Linux GPU/CPU PYTHON 服务化部署测试

Linux GPU/CPU  PYTHON 服务化部署测试的主程序为`test_serving_infer_python.sh`，可以测试基于Python的模型服务化部署功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP-LiteSeg   |  pp_liteSeg_stdc1 |  支持 | 支持 | 1 |
|  PP-LiteSeg   |  pp_liteSeg_stdc2 |  支持 | 支持 | 1 |
|  PP-Humanseg  |  pphumanseg_lite     |  支持 | 支持 | 1 |
|  PP-Humanseg  |  fcn_hrnetw18_small   |  支持 | 支持 | 1 |
|  PP-Humanseg  |  deeplabv3p_resnet50   |  支持 | 支持 | 1 |
|  PP-Matting   |  ppmatting |  支持 | 支持 | 1 |
|  FCN          |  fcn_hrnetw18     |  支持 | 支持 | 1 |
|  OCRNet       |  ocrnet_hrnetw18  |  支持 | 支持 | 1 |
|  OCRNet       |  ocrnet_hrnetw48  |  支持 | 支持 | 1 |
|  STDCSeg      |  stdc_stdc1       |  支持 | 支持 | 1 |



## 2. 测试流程

### 2.1 准备环境

* 配置docker

拉取并进入 Paddle Serving的GPU Docker。

```
nvidia-docker pull registry.baidubce.com/paddlepaddle/serving:0.8.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit registry.baidubce.com/paddlepaddle/serving:0.8.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash
```

* 安装Python

安装Python，支持Python3.6/3.7/3.8/3.9，推荐python3.7。

* 安装 PaddleServing

安装PaddleServing相关组件，包括serving-server、serving_client、serving-app

```
pip3 install paddle-serving-client==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install paddle-serving-app==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU Server，需要确认环境再选择执行哪一条，推荐使用CUDA 10.2的包
pip3 install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

PaddleServing 0.8版本的安装说明，参考[链接](https://github.com/PaddlePaddle/Serving/blob/v0.8.3/doc/Install_CN.md)


* 安装PaddlePaddle

```
# GPU CUDA 10.2环境请执行
pip3 install paddlepaddle-gpu==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 准备PaddleSeg

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip install -r requirements.txt
```

 ### 2.2 功能测试

 测试方法如下所示，希望测试不同的模型文件，只需更换为对应的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_serving_infer_python.sh ${your_params_file} lite_train_lite_infer
```

以`PP-LiteSeg-Stdc1`的`Linux GPU/CPU PYTHON 服务化部署测试`为例，在PaddleSeg根目录下，执行如下命令。

```bash
bash test_tipc/prepare.sh test_tipc/configs/pp_liteseg_stdc1/serving_infer_python.txt serving_infer
```

```bash
bash test_tipc/test_serving_infer_python.sh test_tipc/configs/pp_liteseg_stdc1/serving_infer_python.txt serving_infer
```

输出结果如下，表示命令运行成功。

```
Run successfully with command - python3 ./pipeline_http_client.py --img_path=../data/cityscapes_demo.png > ../../log/pp_liteseg_stdc1/serving_infer/serving_infer_python_gpu_batchsize_1.log 2>&1 !
```

预测结果会自动保存在 `./log/pp_liteseg_stdc1/serving_infer/serving_infer_python_gpu_batchsize_1.log` ，可以看到 PaddleServing 的运行结果（如下），其中不同模型的value数值不同。

```
{'err_no': 0, 'err_msg': '', 'key': ['res_mean_val'], 'value': ['[5.126434326171875]'], 'tensors': []}
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
