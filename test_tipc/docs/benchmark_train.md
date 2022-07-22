
# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

# 1. 测试流程
## 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_infer_python.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/segformer_b0/train_infer_python.txt benchmark_train
```

## 1.2 功能测试
执行`test_tipc/benchmark_train.sh`，完成模型训练和日志解析

```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_infer_python.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/segformer_b0/train_infer_python.txt benchmark_train

```

`test_tipc/benchmark_train.sh`支持根据传入的第三个参数实现只运行某一个训练配置，如下：
```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_infer_python.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/segformer_b0/train_infer_python.txt benchmark_train  dynamic_bs2_fp32_DP_N1C1
```
dynamic_bs2_fp32_DP_N1C1为test_tipc/benchmark_train.sh传入的参数，格式如下：
`${modeltype}_${batch_size}_${fp_item}_${run_mode}_${device_num}`
包含的信息有：模型类型、batchsize大小、训练精度如fp32,fp16等、分布式运行模式以及分布式训练使用的机器信息如单机单卡（N1C1）。


## 2. 日志输出

运行后将保存模型的训练日志和解析日志，使用 `test_tipc/configs/segformer_b0/train_infer_python.txt` 参数文件的训练日志解析结果是：

```
{"model_branch": "dygaph", "model_commit": "7c39a1996b19087737c05d883fd346d2f39dbcc0", "model_name": "segformer_b0_bs2_fp32_SingleP_DP", "batch_size": 8, "fp_item": "fp32", "run_process_type": "SingleP", "run_mode": "DP", "convergence_value": "5.413110", "convergence_key": "loss:", "ips": 19.333, "speed_unit": "samples/s", "device_num": "N1C1", "model_run_time": "0", "frame_commit": "8cc09552473b842c651ead3b9848d41827a3dbab", "frame_version": "0.0.0"}
```

训练日志和日志解析结果保存在benchmark_log目录下，文件组织格式如下：
```
train_log/
├── index
│   ├── PaddleSeg_segformer_b0_bs2_fp32_SingleP_DP_N1C1_speed
│   └── PaddleSeg_segformer_b0_bs2_fp32_SingleP_DP_N1C4_speed
├── profiling_log
│   └── PaddleSeg_segformer_b0_bs2_fp32_SingleP_DP_N1C1_profiling
└── train_log
    ├── PaddleSeg_segformer_b0_bs2_fp32_SingleP_DP_N1C1_log
    └── PaddleSeg_segformer_b0_bs2_fp32_SingleP_DP_N1C4_log
```

## 3. 各模型单卡性能数据一览

|模型名称|配置文件|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|7.206|7.163|7.194|0.006|
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|18.557|18.308|18.610|0.016|
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|||||
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|19.931|19.790|19.573|0.018|
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|7.306|7.267|7.248|0.008|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4.192|4.204|4.205|0.003|
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|4.221|4.197|4.203|0.006|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|||||
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|||||

*注：以上速度指标均在单卡（1块Nvidia V100 GPU）、不使用混合精度的情况下测得，测试时使用的batch size为对应配置中batch size候选项中的最小值。*
