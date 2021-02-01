# PaddleSeg Benchmark with AMP

## 动态图

通过 **--fp16** 开启amp训练。

单机单卡使用如下命令进行训练：
```
export CUDA_VISIBLE_DEVICES=0
python train.py --config benchmark/hrnet.yml --iters 2000 --log_iters 10 --fp16
```

单机多卡使用如下命令进行训练：
```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch train.py --config benchmark/hrnet.yml --iters 2000 --log_iters 10 --fp16
```

## 静态图

通过 **MODEL.FP16 True** 开启amp训练
单机单卡使用如下命令进行训练：
```
cd legacy
export CUDA_VISIBLE_DEVICES=0
python pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 2 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

单机单卡使用如下命令进行训练：
```
export CUDA_VISIBLE_DEVICES=0,1
python pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 4 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

## 竞品
竞品为[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
对应竞品配置文件为：configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py
相关执行方式其参考官方仓库。
