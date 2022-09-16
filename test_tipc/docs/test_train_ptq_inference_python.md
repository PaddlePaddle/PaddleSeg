# Linux GPU/CPU 离线量化训练推理测试

Linux GPU/CPU 离线量化训练推理测试的主程序为`test_ptq_inference_python.sh`。

## 1. 测试结论汇总

| 算法名称 | 模型名称 | 单机单卡 |
|  :----: |   :----:  |    :----:  |  
|  HRNet  | pphumanseg_mobile (fcn_hrnetw18_small) | KL离线量化（无需训练） |
|  HRNet  | fcn_hrnetw18 | KL离线量化（无需训练） |
|  DeepLabV3P  | pphumanseg_server (deeplabv3p_resnet50) | KL离线量化（无需训练） |
|  ConnectNet  | pphumanseg_lite | KL离线量化（无需训练） |
|  OCRNet  | ocrnet_hrnetw18 | KL离线量化（无需训练） |
|  OCRNet  | ocrnet_hrnetw48 | KL离线量化（无需训练） |
|  SegFormer  | segformer_b0 | KL离线量化（无需训练） |
|  PP-LiteSeg  | pp_liteseg_stdc1 | KL离线量化（无需训练） |
|  PP-LiteSeg  | pp_liteseg_stdc2 | KL离线量化（无需训练） |


## 2. 测试流程

### 2.1 准备数据和模型

准备离线量化需要的模型和数据。以`fcn_hrnetw18`为例，执行如下指令：

```bash
bash test_tipc/prepare.sh test_tipc/configs/fcn_hrnetw18/train_ptq_infer_python.txt whole_infer
```

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip3 install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip3 install paddlepaddle==2.2.0
    ```
- 安装PaddleSlim
    ```
    pip3 install paddleslim==2.2.0
    ```
- 安装依赖
    ```
    pip3 install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip3 install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```


### 2.3 功能测试

以`fcn_hrnetw18`的`Linux GPU/CPU KL离线量化训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_ptq_inference_python.sh test_tipc/configs/fcn_hrnetw18/train_ptq_infer_python.txt whole_infer
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command - fcn_hrnetw18_KL - python3.7 deploy/slim/quant/ptq.py   --model_dir=test_tipc/output/fcn_hrnetw18_KL/fcn_hrnetw18_cityscapes_1024x512_80k --batch_num=1 --batch_size=1 --config=test_tipc/configs/fcn_hrnetw18/fcn_hrnetw18_1024x512_cityscapes.yml   >./test_tipc/output/fcn_hrnetw18_KL/whole_infer/export.log 2>&1 - ./test_tipc/output/fcn_hrnetw18_KL/whole_infer/export.log
 ......
 Run successfully with command - fcn_hrnetw18_KL - python3.7 deploy/python/infer.py --device=cpu --config=quant_model/deploy.yaml --batch_size=1 --image_path=test_tipc/cpp/cityscapes_demo.png --benchmark=True > ./test_tipc/output/fcn_hrnetw18_KL/whole_infer/python_infer_cpu_batchsize_1.log 2>&1  - ./test_tipc/output/fcn_hrnetw18_KL/whole_infer/python_infer_cpu_batchsize_1.log
 ......
 Run successfully with command - fcn_hrnetw18_KL - python3.7 deploy/python/infer.py --device=gpu --config=quant_model/deploy.yaml --batch_size=1 --image_path=test_tipc/cpp/cityscapes_demo.png --benchmark=True > ./test_tipc/output/fcn_hrnetw18_KL/whole_infer/python_infer_gpu_batchsize_1.log 2>&1  - ./test_tipc/output/fcn_hrnetw18_KL/whole_infer/python_infer_gpu_batchsize_1.log
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
