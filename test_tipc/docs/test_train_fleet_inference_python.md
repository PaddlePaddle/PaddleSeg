# Linux GPU/CPU 多机多卡训练推理测试

Linux GPU/CPU 多机多卡训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 多机多卡 |
|  :----: |   :----:  |    :----:  |
|  pp_liteseg    | pp_liteseg_stdc1     | 分布式训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  pp_liteseg  |  pp_liteseg_stdc1 |  支持 | 支持 | 1 |


## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 功能测试

#### 2.1.1 修改配置文件

首先，修改配置文件中的`ip`设置:  假设两台机器的`ip`地址分别为`192.168.0.1`和`192.168.0.2`，则对应的配置文件`gpu_list`字段需要修改为`gpu_list:192.168.0.1,192.168.0.2;0,1`； `ip`地址查看命令为`ifconfig`。


#### 2.1.2 准备数据

运行`prepare.sh`准备数据和模型，以配置文件`test_tipc/configs/pp_liteseg_stdc1/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt`为例，数据准备命令如下所示。

```shell
bash test_tipc/prepare.sh test_tipc/configs/pp_liteseg_stdc1/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

**注意：** 由于是多机训练，这里需要在所有的节点上均运行启动上述命令，准备数据。

#### 2.1.3 修改起始端口并开始测试

在多机的节点上使用下面的命令设置分布式的起始端口（否则后面运行的时候会由于无法找到运行端口而hang住），一般建议设置在`10000~20000`之间。

```shell
export FLAGS_START_PORT=17000
```

以配置文件`test_tipc/configs/pp_liteseg_stdc1/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt`为例，测试方法如下所示。

```shell
bash test_tipc/test_train_inference_python.sh  test_tipc/configs/pp_liteseg_stdc1/train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

**注意：** 由于是多机训练，这里需要在所有的节点上均运行启动上述命令进行测试。


#### 2.1.4 输出结果

输出结果如下，表示命令运行成功。

```bash
[33m Run successfully with command - pp_liteseg_stdc1 - python3.7 -m paddle.distributed.launch --ips=192.168.0.1,192.168.0.2 --gpus=0,1 train.py
--config test_tipc/configs/pp_liteseg_stdc1/pp_liteseg_stdc1_cityscapes_1024x512_160k.yml --do_eval --save_interval 500 --seed 100   --save_dir=./test_
tipc/output/pp_liteseg_stdc1/norm_gpus_0,1_autocast_null_nodes_2   --iters=20   --batch_size=2    !  
 ......
[33m Run successfully with command - pp_liteseg_stdc1 - python3.7 deploy/python/infer.py --save_dir ./test_tipc/output/pp_liteseg_stdc1/ --device=gpu
 --use_trt=False --precision=fp32 --config=./test_tipc/output/pp_liteseg_stdc1/norm_gpus_0,1_autocast_null_nodes_2//deploy.yaml --batch_size=1 --image_
path=test_tipc/data/cityscapes/cityscapes_val_5.list --benchmark=True   > ./test_tipc/output/pp_liteseg_stdc1/python_infer_gpu_usetrt_False_precision_f
p32_batchsize_1.log 2>&1 !  ^[[0m
```

在开启benchmark参数时，可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU,GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```
2022-06-09 21:14:56 [INFO]      ---------------------- Env info ----------------------  
2022-06-09 21:14:56 [INFO]       OS_version: CentOS 6.10  
2022-06-09 21:14:56 [INFO]       CUDA_version: 10.1.243  
2022-06-09 21:14:56 [INFO]       CUDNN_version: None.None.None  
2022-06-09 21:14:56 [INFO]       drivier_version: 460.32.03  
2022-06-09 21:14:56 [INFO]      ---------------------- Paddle info ----------------------  
2022-06-09 21:14:56 [INFO]       paddle_version: 2.3.0-rc0  
2022-06-09 21:14:56 [INFO]       paddle_commit: 5d4980c052583fec022812d9c29460aff7cdc18b  
2022-06-09 21:14:56 [INFO]       log_api_version: 1.0  
2022-06-09 21:14:56 [INFO]      ----------------------- Conf info -----------------------  
2022-06-09 21:14:56 [INFO]       runtime_device: gpu  
2022-06-09 21:14:56 [INFO]       ir_optim: True  
2022-06-09 21:14:56 [INFO]       enable_memory_optim: True  
2022-06-09 21:14:56 [INFO]       enable_tensorrt: False  
2022-06-09 21:14:56 [INFO]       enable_mkldnn: False  
2022-06-09 21:14:56 [INFO]       cpu_math_library_num_threads: 1  
2022-06-09 21:14:56 [INFO]      ----------------------- Model info ----------------------  
2022-06-09 21:14:56 [INFO]       model_name:  
2022-06-09 21:14:56 [INFO]       precision: fp32  
2022-06-09 21:14:56 [INFO]      ----------------------- Data info -----------------------  
2022-06-09 21:14:56 [INFO]       batch_size: 1  
2022-06-09 21:14:56 [INFO]       input_shape: dynamic  
2022-06-09 21:14:56 [INFO]       data_num: 5  
2022-06-09 21:14:56 [INFO]      ----------------------- Perf info -----------------------  
2022-06-09 21:14:56 [INFO]       cpu_rss(MB): 2934.3672, gpu_rss(MB): 1162.0, gpu_util: 3.0%  
2022-06-09 21:14:56 [INFO]       total time spent(s): 0.4089  
2022-06-09 21:14:56 [INFO]       preprocess_time(ms): 75.9658, inference_time(ms): 4.854, postprocess_time(ms): 0.9655
```

该信息可以在运行log中查看，以上面的`pp_liteseg_stdc1`为例，log位置在`./test_tipc/output/pp_liteseg_stdc1/results_python.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。

**注意：** 由于分布式训练时，仅在`trainer_id=0`所在的节点中保存模型，因此如果测试多机的推理过程，其他的节点中在运行模型导出与推理时会报错，为正常现象。
