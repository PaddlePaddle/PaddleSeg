
# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

## 1. 测试流程
### 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_infer_python.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/segformer_b0/train_infer_python.txt benchmark_train
```

### 1.2 功能测试
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

训练日志和日志解析结果保存在`${BENCHMARK_LOG_DIR}`目录下，文件组织格式如下：
```
${BENCHMARK_LOG_DIR}/
├── index
│   ├── PaddleSeg_segformer_b0_bs2_fp32_DP_N1C1_speed
│   └── PaddleSeg_segformer_b0_bs2_fp32_DP_N1C8_speed
├── profiling_log
│   └── PaddleSeg_segformer_b0_bs2_fp32_DP_N1C1_profiling
└── train_log
    ├── PaddleSeg_segformer_b0_bs2_fp32_DP_N1C1_log
    └── PaddleSeg_segformer_b0_bs2_fp32_DP_N1C8_log
```

## 3. 各模型单卡性能数据一览

*注：本节中的速度指标均使用单卡（1块Nvidia V100 GPU）测得。通常情况下，测试时使用对应配置中的batch_size候选项；对于HRNet_W18，由于测试机器的显存限制，选取batch_size=4。*

### 3.1 大数据集+fp32精度

|模型名称|配置文件|batch size|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|-|-|-|-|
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|-|-|-|-|
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|-|-|-|-|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|-|-|-|-|
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|||||
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|||||
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|||||
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|||||
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|3.934|3.945|3.980|0.012|
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|3.956|3.989|4.001|0.011|

### 3.2 大数据集+fp16精度

|模型名称|配置文件|batch size|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|-|-|-|-|
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|-|-|-|-|
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|-|-|-|-|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|-|-|-|-|
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|||||
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|||||
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|||||
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|||||
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|3.814|3.826|3.826|0.003|
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|3.971|3.953|3.985|0.008|
|SFNet|[config](./configs/sfnet/train_infer_python.txt)|4|||||
|MobileSeg-MV3|[config](./configs/mobileseg_mv3/train_infer_python.txt)|4|||||

### 3.3 小数据集+fp32精度

|模型名称|配置文件|batch size|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|7.206|7.163|7.194|0.006|
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|18.557|18.308|18.610|0.016|
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|1.863|1.809|1.835|0.029|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|19.931|19.790|19.573|0.018|
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|12.047|11.982|11.847|0.017|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|22.021|22.897|25.484|0.136|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|24.335|24.527|21.466|0.125|
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|3.869|3.948|3.930|0.020|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|9.152|9.150|9.247|0.010|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|3.948|3.924|3.987|0.016|
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|3.942|3.972|3.921|0.013|
|SFNet|[config](./configs/sfnet/train_infer_python.txt)|4|6.350|6.340|6.308|0.007|
|MobileSeg-MV3|[config](./configs/mobileseg_mv3/train_infer_python.txt)|4|17.014|17.466|17.453|0.026|

### 3.4 小数据集+fp16精度

|模型名称|配置文件|batch size|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|9.629|9.582|9.553|0.008|
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|18.432|18.597|18.712|0.015|
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|1.832|1.829|1.838|0.005|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|20.029|19.911|20.048|0.007|
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|11.025|11.248|11.187|0.020|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|22.392|25.139|24.049|0.109|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|22.547|23.748|21.415|0.098|
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|3.966|3.971|3.947|0.006|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|13.659|13.449|13.688|0.017|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|3.969|3.938|3.953|0.008|
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|4.098|4.085|4.105|0.005|
|SFNet|[config](./configs/sfnet/train_infer_python.txt)|4|10.732|10.690|10.642|0.008|
|MobileSeg-MV3|[config](./configs/mobileseg_mv3/train_infer_python.txt)|4|16.350|16.347|16.336|0.001|

## 4. 各模型多卡性能数据一览

### 4.1 8卡+小数据集+fp32精度

|模型名称|配置文件|batch size per GPU|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|||||
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|||||
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|10.945|11.180|11.260|0.028|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|||||
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|60.094|60.500|60.477|0.007|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|67.234|67.591|67.532|0.005|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|55.381|55.508|55.655|0.005|
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|26.562|26.495|26.103|0.017|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|50.094|50.166|50.173|0.002|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|||||
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|||||
|SFNet|[config](./configs/sfnet/train_infer_python.txt)|4|43.261|43.254|43.447|0.004|
|MobileSeg-MV3|[config](./configs/mobileseg_mv3/train_infer_python.txt)|4|88.654|87.520|87.055|0.018|

### 4.2 8卡+小数据集+fp16精度

|模型名称|配置文件|batch size per GPU|第1次测试FPS`fps_1`|第2次测试FPS`fps_2`|第3次测试FPS`fps_3`|`(max(fps_n)-min(fps_n))/max(fps_n)`|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-HumanSeg-Server|[config](./configs/deeplabv3p_resnet50/train_infer_python.txt)|2|||||
|PP-HumanSeg-Lite|[config](./configs/pphumanseg_lite/train_infer_python.txt)|2|||||
|PP-Matting|[config](./configs/ppmatting/train_infer_python.txt)|2|11.183|11.209|11.485|0.026|
|PP-HumanSeg-Mobile|[config](./configs/fcn_hrnetw18_small/train_infer_python.txt)|2|||||
|HRNet_W18|[config](./configs/fcn_hrnetw18/train_infer_python.txt)|4|50.589|51.518|51.376|0.018|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|2|63.272|62.597|63.258|0.011|
|Fast-SCNN|[config](./configs/fastscnn/train_infer_python.txt)|4|55.406|55.575|55.174|0.007|
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|2|||||
|OCRNet_HRNetW48|[config](./configs/ocrnet_hrnetw48/ocrnet_hrnetw48_cityscapes_1024x512.yml)|4|||||
|OCRNet_HRNetW18|[config](./configs/ocrnet_hrnetw18/train_infer_python.txt)|2|25.742|25.211|25.369|0.021|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|2|56.570|55.315|55.091|0.026|
|SegFormer_B0|[config](./configs/segformer_b0/train_infer_python.txt)|4|||||
|PP-LiteSeg-T|[config](./configs/pp_liteseg_stdc1/train_infer_python.txt)|2|||||
|PP-LiteSeg-B|[config](./configs/pp_liteseg_stdc2/train_infer_python.txt)|2|||||
|SFNet|[config](./configs/sfnet/train_infer_python.txt)|4|58.499|58.387|58.219|0.005|
|MobileSeg-MV3|[config](./configs/mobileseg_mv3/train_infer_python.txt)|4|82.415|81.367|82.672|0.016|
