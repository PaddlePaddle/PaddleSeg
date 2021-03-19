# PaddleSeg Benchmark with AMP

## 动态图
数据集cityscapes 放置于data目录下, 下载链接：https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar

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
# fleet开启多卡训练
fleetrun train.py --config benchmark/hrnet.yml --iters 2000 --log_iters 10 --fp16
```

DeepLabv3+ 模型的配置文件为：
benchmark/deeplabv3p.yml

**注意**

* 动态图中batch_size设置为每卡的batch_size
* DeepLabv3+ 支持通过传入 **--data_format NHWC**进行‘NHWC’数据格式的训练。



## 静态图
数据集cityscapes 放置于legacy/dataset目录下

通过 **MODEL.FP16 True** 开启amp训练
单机单卡使用如下命令进行训练：
```
cd legacy
export CUDA_VISIBLE_DEVICES=0
python pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 2 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

单机多卡使用如下命令进行训练：
```
export CUDA_VISIBLE_DEVICES=0,1
fleetrun pdseg/train.py --cfg configs/hrnetw18_cityscapes_1024x512_215.yaml --use_gpu  --use_mpio --log_steps 10 BATCH_SIZE 4 SOLVER.NUM_EPOCHS 3 MODEL.FP16 True
```

deeplabv3p模型的配置文件为：
configs/deeplabv3p_resnet50_vd_cityscapes.yaml

**注意**
静态图中的BATCH_SIZE为总的batch size。

## 竞品
竞品为[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

对应竞品配置文件为：

configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py

configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py

相关执行方式请参考其官方仓库。
