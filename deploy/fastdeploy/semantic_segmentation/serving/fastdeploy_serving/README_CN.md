[English](README.md) | 简体中文
# PaddleSeg 服务化部署示例

PaddleSeg 服务化部署示例是利用FastDeploy Serving搭建的服务化部署示例。FastDeploy Serving是基于Triton Inference Server框架封装的适用于高并发、高吞吐量请求的服务化部署框架，是一套可用于实际生产的完备且性能卓越的服务化部署框架。如没有高并发，高吞吐场景的需求，只想快速检验模型线上部署的可行性，请参考[fastdeploy_serving](../simple_serving/)

## 1. 部署环境准备
在服务化部署前，需确认服务化镜像的软硬件环境要求和镜像拉取命令，请参考[FastDeploy服务化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)


## 2. 启动服务

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/fastdeploy_serving

# 下载PP-LiteSeg模型文件
wget  https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# 将模型文件放入 models/runtime/1目录下
mv PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer/model.pdmodel models/runtime/1/
mv PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer/model.pdiparams models/runtime/1/

# 拉取fastdeploy镜像(x.y.z为镜像版本号，需参照serving文档替换为数字)
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /serving 目录
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# 启动服务(不设置CUDA_VISIBLE_DEVICES环境变量，会拥有所有GPU卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```
**注意**: 当出现"Address already in use", 请使用`--grpc-port`指定端口号来启动服务，同时更改paddleseg_grpc_client.py中的请求端口号

服务启动成功后， 会有以下输出:
```bash
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## 3. 客户端请求

在物理机器中执行以下命令，发送grpc请求并输出结果
```bash
# 下载测试图片
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 安装客户端依赖
python3 -m pip install tritonclient[all]

# 发送请求
python3 paddleseg_grpc_client.py
```

发送请求成功后，会返回json格式的检测结果并打印输出:
```bash
tm: name: "INPUT"
datatype: "UINT8"
shape: -1
shape: -1
shape: -1
shape: 3

output_name: SEG_RESULT
Only print the first 20 labels in label_map of SEG_RESULT
{'label_map': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'score_map': [], 'shape': [1024, 2048], 'contain_score_map': False}
```

## 4. 配置修改

当前默认配置在CPU上运行ONNXRuntime引擎， 如果要在GPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md)

## 5. 更多指南
- [使用 VisualDL 进行 Serving 可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md)

## 6. 常见问题
- [如何编写客户端 HTTP/GRPC 请求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md)
- [如何编译服务化部署镜像](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/compile.md)
- [服务化部署原理及动态Batch介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/demo.md)
- [模型仓库介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_repository.md)
