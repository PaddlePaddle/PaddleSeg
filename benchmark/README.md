English | [简体中文](README_CN.md)

# Benchmark Introduction

The content is as follow:

```
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
```

## Environment
Use Docker to configure the environment.
* docker image: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
* CUDA 10.1 + cudnn7
* paddle=2.1.2
* py=37

## Test
### Training Test

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
bash benchmark/run_all.sh
```
### How to Open Profiling
 Add the following parameter when training.
 `--profiler_options="batch_range=[50, 60]; profile_path=model.profile`
