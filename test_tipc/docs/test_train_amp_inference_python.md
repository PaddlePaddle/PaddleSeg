# Linux GPU/CPU 混合精度训练推理测试

Linux GPU/CPU 混合精度训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  PP_LiteSeg   |  pp_liteseg_stdc1 |  混合精度训练 | 混合精度训练 |
|  PP_LiteSeg   |  pp_liteseg_stdc2 |  混合精度训练 | 混合精度训练 |
|  ConnectNet   |  pp_humanseg_lite | 混合精度训练 | 混合精度训练 |
|  HRNet W18 Small   | pp_humanseg_mobile  |  混合精度训练 | 混合精度训练 |
|  DeepLabV3P ResNet50   |  pp_humanseg_server |  混合精度训练 | 混合精度训练 |
|  HRNet   |  fcn_hrnet_w18 |  混合精度训练 | 混合精度训练 |
|  OCRNet   |  ocrnet_hrnetw18 |  混合精度训练 | 混合精度训练 |
|  OCRNet   |  ocrnet_hrnetw48 |  混合精度训练 | 混合精度训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  PP_LiteSeg   |  pp_liteseg_stdc1 |  支持 | 支持 | 1 |
|  PP_LiteSeg   |  pp_liteseg_stdc2 |  支持 | 支持 | 1 |
|  ConnectNet   |  pp_humanseg_lite |  支持 | 支持 | 1 |
|  HRNet W18 Small   | pp_humanseg_mobile  |  支持 | 支持 | 1 |
|  DeepLabV3P ResNet50    |  pp_humanseg_server |  支持 | 支持 | 1 |
|  HRNet   |  fcn_hrnet_w18 |  支持 | 支持 | 1 |
|  OCRNet   |  ocrnet_hrnetw18 |  支持 | 支持 | 1 |
|  OCRNet   |  ocrnet_hrnetw48 |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.2
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.2
    ```

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.2 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`pphumanseg_lite`的`Linux GPU/CPU 混合精度训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/pphumanseg_lite/train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/pphumanseg_lite/train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
[33m Run successfully with command - python3.7 train.py --config test_tipc/configs/pphumanseg_lite/pphumanseg_lite_mini_supervisely.yml --precision fp16 --amp_level O2 --do_eval --save_interval 500 --seed 100    --save_dir=./test_tipc/output/pphumanseg_lite/amp_train_gpus_0_autocast_null --iters=50     --batch_size=2  
......
[33m Run successfully with command - python3.7 deploy/python/infer.py --device=cpu --enable_mkldnn=True --cpu_threads=1 --config=./test_tipc/output/pphumanseg_lite/amp_train_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/mini_supervisely/test.txt --benchmark=True --precision=fp32 --save_dir=./test_tipc/output/pphumanseg_lite/python_infer_cpu_usemkldnn_True_threads_1_precision_fp32_batchsize_1_results   > ./test_tipc/output/pphumanseg_lite/python_infer_cpu_usemkldnn_True_threads_1_precision_fp32_batchsize_1.log 2>&1 !
```

在开启benchmark选项时，可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU/GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```
2022-04-13 11:43:09 [INFO]    ---------------------- Env info ----------------------
2022-04-13 11:43:09 [INFO]     OS_version: CentOS Linux 7
2022-04-13 11:43:09 [INFO]     CUDA_version: 11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
2022-04-13 11:43:09 [INFO]     CUDNN_version: None.None.None
2022-04-13 11:43:09 [INFO]     drivier_version: 460.27.04
2022-04-13 11:43:09 [INFO]    ---------------------- Paddle info ----------------------
2022-04-13 11:43:09 [INFO]     paddle_version: 2.2.2
2022-04-13 11:43:09 [INFO]     paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
2022-04-13 11:43:09 [INFO]     log_api_version: 1.0
2022-04-13 11:43:09 [INFO]    ----------------------- Conf info -----------------------
2022-04-13 11:43:09 [INFO]     runtime_device: cpu
2022-04-13 11:43:09 [INFO]     ir_optim: True
2022-04-13 11:43:09 [INFO]     enable_memory_optim: True
2022-04-13 11:43:09 [INFO]     enable_tensorrt: False
2022-04-13 11:43:09 [INFO]     enable_mkldnn: False
2022-04-13 11:43:09 [INFO]     cpu_math_library_num_threads: 1
2022-04-13 11:43:09 [INFO]    ----------------------- Model info ----------------------
2022-04-13 11:43:09 [INFO]     model_name:
2022-04-13 11:43:09 [INFO]     precision: fp32
2022-04-13 11:43:09 [INFO]    ----------------------- Data info -----------------------
2022-04-13 11:43:09 [INFO]     batch_size: 1
2022-04-13 11:43:09 [INFO]     input_shape: dynamic
2022-04-13 11:43:09 [INFO]     data_num: 50
2022-04-13 11:43:09 [INFO]    ----------------------- Perf info -----------------------
2022-04-13 11:43:09 [INFO]     cpu_rss(MB): 315.2656, gpu_rss(MB): 8842.0, gpu_util: 0.0%
2022-04-13 11:43:09 [INFO]     total time spent(s): 4.2386
2022-04-13 11:43:09 [INFO]     preprocess_time(ms): 33.7887, inference_time(ms): 50.7443, postprocess_time(ms): 0.2391
```

该信息可以在运行log中查看，以`pphumanseg_lite`为例，log位置在`./output/pphumanseg_lite/lite_train_lite_infer/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

`注意`: 混合精度参数配置文件中，默认使用O1模式；O2模式存在部分问题，需要安装PaddlePaddle develop版本才可使用。
