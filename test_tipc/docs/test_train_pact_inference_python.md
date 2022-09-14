# Linux GPU/CPU PACT量化训练推理测试

Linux GPU/CPU PACT量化训练推理测试的主程序为`test_train_inference_python.sh`。

## 1. 测试结论汇总

| 算法名称 | 模型名称 | 单机单卡 |
|  :----: |   :----:  |    :----:  |  
|  HRNet  | pphumanseg_mobile (fcn_hrnetw18_small) | PACT量化训练 |
|  HRNet  | fcn_hrnetw18 | PACT量化训练 |
|  DeepLabV3P  | pphumanseg_server (deeplabv3p_resnet50) | PACT量化训练 |
|  ConnectNet  | pphumanseg_lite | PACT量化训练 |
|  OCRNet  | ocrnet_hrnetw18 | PACT量化训练 |
|  OCRNet  | ocrnet_hrnetw48 | PACT量化训练 |
|  SegFormer  | segformer_b0 | PACT量化训练 |
|  PP-LiteSeg  | pp_liteseg_stdc1 | PACT量化训练 |
|  PP-LiteSeg  | pp_liteseg_stdc2 | PACT量化训练 |


## 2. 测试流程

### 2.1 准备数据和模型

准备PACT量化训练推理测试需要的模型和数据。以`fcn_hrnetw18`为例，执行如下指令：

```bash
bash test_tipc/prepare.sh test_tipc/configs/fcn_hrnetw18/train_pact_infer_python.txt lite_train_lite_infer
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

以`fcn_hrnetw18`的`Linux GPU/CPU PACT量化训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/fcn_hrnetw18/train_pact_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command - fcn_hrnetw18 - python3.7 slim/quant/qat_train.py --config test_tipc/configs/fcn_hrnetw18/fcn_hrnetw18_1024x512_cityscapes.yml --device gpu --save_interval 500 --seed 100 --num_workers 8    --save_dir=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null --iters=20    --batch_size=2     - ./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null_nodes_1.log
 ......
 Run successfully with command - fcn_hrnetw18 - python3.7 slim/quant/qat_val.py --config test_tipc/configs/fcn_hrnetw18/fcn_hrnetw18_1024x512_cityscapes.yml --device gpu --num_workers 8 --model_path=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null/iter_20/model.pdparams     - ./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null_nodes_1_eval.log
 ......
 Run successfully with command - fcn_hrnetw18 - python3.7 slim/quant/qat_export.py --config test_tipc/configs/fcn_hrnetw18/fcn_hrnetw18_1024x512_cityscapes.yml --model_path=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null/iter_20/model.pdparams --save_dir=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null - ./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null_nodes_1_export.log
 ......
 Run successfully with command - fcn_hrnetw18 - python3.7 deploy/python/infer.py --device=cpu --enable_mkldnn=True --cpu_threads=6 --config=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/pact_train_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/cityscapes/cityscapes_val_5.list --benchmark=True --precision=int8 --save_dir=./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/python_infer_cpu_usemkldnn_True_threads_6_precision_int8_batchsize_1_results --model_name=fcn_hrnetw18 > ./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/python_infer_cpu_gpu_0_usemkldnn_True_threads_6_precision_int8_batchsize_1.log 2>&1  - ./test_tipc/output/fcn_hrnetw18/lite_train_lite_infer/python_infer_cpu_gpu_0_usemkldnn_True_threads_6_precision_int8_batchsize_1.log
 ......
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
