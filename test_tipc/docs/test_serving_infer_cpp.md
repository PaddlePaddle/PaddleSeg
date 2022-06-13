# Linux GPU/CPU CPP 服务化部署测试

Linux GPU/CPU CPP 服务化部署测试的主程序为`test_serving_infer_cpp.sh`，可以测试基于CPP的模型服务化部署功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP-LiteSeg   |  pp-liteSeg-stdc1 |  支持 | 支持 | 1 |
|  PP-LiteSeg   |  pp-liteSeg-stdc2 |  支持 | 支持 | 1 |
|  PP-Humanseg  |  pphumanseg_lite     |  支持 | 支持 | 1 |
|  PP-Humanseg  |  pphumanseg_mobile   |  支持 | 支持 | 1 |
|  PP-Humanseg  |  pphumanseg_server   |  支持 | 支持 | 1 |
|  PP-Matting   |  pp_humanseg_matting |  支持 | 支持 | 1 |
|  HRNet        |  fcn_hrnetw18     |  支持 | 支持 | 1 |
|  OCRNet       |  ocrnet_hrnetw18  |  支持 | 支持 | 1 |
|  OCRNet       |  ocrnet_hrnetw48  |  支持 | 支持 | 1 |
|  STDCSeg      |  stdc_stdc1       |  支持 | 支持 | 1 |



## 2. 测试流程

### 2.1 准备环境

* 配置docker

拉取并进入Paddle Serving的GPU Docker (serving0.8.0-cuda10.2-cudnn7)。

```
nvidia-docker pull registry.baidubce.com/paddlepaddle/serving:0.8.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test_serving_cpp -dit registry.baidubce.com/paddlepaddle/serving:0.8.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test_serving_cpp bash
```

* 安装 PaddleServing

安装PaddleServing最新版本的相关组件，包括serving_client、serving-app。
PaddleServing的详细安装说明，参考[链接](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/doc/Install_CN.md)。

比如执行下面命令，在GPU CUDA 10.2环境下，安装0.8.3版本的PaddleServing组件。

```
pip3.7 install paddle-serving-client==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3.7 install paddle-serving-app==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 安装PaddlePaddle

安装最新版本的PaddlePaddle，具体参考[文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

比如执行下面命令，在GPU CUDA 10.2环境下，安装2.3.0版本PaddlePaddle。

```
pip3.7 install paddlepaddle-gpu==2.3.0 -i https://mirror.baidu.com/pypi/simple
```

* 准备PaddleSeg

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
git checkout -b develop origin/develop
```

### 2.2 编译安装paddle-serving-server

在PaddleSeg根目录下，执行如下命令编译安装paddle-serving-server，耗时会较长。

```
# export HTTP_PROXY=xxx && export HTTPS_PROXY=xxx
sh test_tipc/serving_cpp/prepare_server.sh
```

在PaddleSeg根目录下，执行如下命令，设置环境变量。
```
export SERVING_BIN=${PWD}/Serving/build_server_gpu_opencv_seg/core/general-server/serving
```

### 2.2 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为对应的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/prepare.sh ${your_params_file} serving_infer
bash test_tipc/test_serving_infer_cpp.sh ${your_params_file} serving_infer
```

以`PP-LiteSeg-Stdc1`的`Linux GPU/CPU CPP 服务化部署测试`为例，在PaddleSeg根目录下，执行如下命令。

```bash
bash test_tipc/prepare.sh test_tipc/configs/pp_liteseg_stdc1/model_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt serving_infer
bash test_tipc/test_serving_infer_cpp.sh test_tipc/configs/pp_liteseg_stdc1/model_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt serving_infer
```

log输出如下，表示测试成功。

```
Run successfully with command - pp_liteseg_stdc1 - python3.7 serving_client.py --input_name=x --output_name=argmax_0.tmp_0 > ../../log/pp_liteseg_stdc1/serving_infer/servering_infer_cpp_gpu_batchsize_1.log 2>&1 !

...

Run successfully with command - pp_liteseg_stdc1 - python3.7 serving_client.py --input_name=x --output_name=argmax_0.tmp_0 > ../../log/pp_liteseg_stdc1/serving_infer/servering_infer_cpp_cpu_batchsize_1.log 2>&1 !
```

GPU预测结果会自动保存在 `./log/pp_liteseg_stdc1/serving_infer/servering_infer_cpp_gpu_batchsize_1.log` ，CPU预测结果会保存在 `./log/pp_liteseg_stdc1/serving_infer/servering_infer_cpp_cpu_batchsize_1.log`，可以看到保存文件中有预测结果的均值（不同模型的value数值不同），表示测试成功。

```
result: {'res_mean_val': '[5.126434326171875]'}
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
