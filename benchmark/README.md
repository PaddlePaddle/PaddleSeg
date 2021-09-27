使用Docker配置Paddle的GPU环境。
* docker image: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
* CUDA 10.1 + cudnn7
* paddle=2.1.2
* py=37

执行训练Benchmark测试。
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
bash benchmark/run_all.sh
```
