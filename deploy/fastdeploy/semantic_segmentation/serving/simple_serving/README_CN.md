简体中文 | [English](README.md)

# PaddleSeg Python轻量服务化部署示例

PaddleSeg Python轻量服务化部署是FastDeploy基于Flask框架搭建的可快速验证线上模型部署可行性的服务化部署示例，基于http请求完成AI推理任务，适用于无并发推理的简单场景，如有高并发，高吞吐场景的需求请参考[fastdeploy_serving](../fastdeploy_serving/)

## 1. 部署环境准备

在部署前，需确认软硬件环境，同时下载预编译python wheel 包，参考文档[FastDeploy预编译库安装](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)

## 2. 启动服务
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
# 注意：如果当前分支找不到测试代码 请切换到develop分支
git checkout develop
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/serving/simple_serving

# 下载PP-LiteSeg模型文件
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# 启动服务，可修改server.py中的配置项来指定硬件、后端等
# 可通过--host、--port指定IP和端口号
fastdeploy simple_serving --app server:app
```

## 3. 客户端请求
```bash
# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 请求服务，获取推理结果（如有必要，请修改脚本中的IP和端口号）
python client.py
```
