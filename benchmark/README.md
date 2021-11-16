# PaddleSeg 下benchmark模型执行说明

├── README.md  
├── configs  
│   ├── cityscapes_30imgs.yml  
│   ├── fastscnn.yml  
│   ├── ocrnet_hrnetw48.yml  
│   └── segformer_b0.yml  
├── deeplabv3p.yml  
├── hrnet.yml  
├── hrnet48.yml  
├── run_all.sh  
├── run_benchmark.sh  
├── run_fp16.sh  
└── run_fp32.sh  
## 环境
使用Docker配置Paddle的GPU环境。
* docker image: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
* CUDA 10.1 + cudnn7
* paddle=2.1.2
* py=37
## 测试步骤
### 执行训练Benchmark测试

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
bash benchmark/run_all.sh
```
### Profiling开关使用方式
```bash
 # 调用train.py时加上该参数 --profiler_options="batch_range=[50, 60]; profile_path=model.profile"
```




