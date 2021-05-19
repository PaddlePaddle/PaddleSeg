# PaddleSeg 多进程训练和混合精度训练

### 环境要求
* PaddlePaddle >= 1.6.1
* NVIDIA NCCL >= 2.4.7

环境配置，数据，预训练模型准备等工作请参考[PaddleSeg使用说明](./usage.md)

### 多进程训练示例

多进程训练，可以按如下方式启动
```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch pdseg/train.py --use_gpu \
                      --do_eval \
                      --cfg configs/unet_pet.yaml \
                      BATCH_SIZE 4 \
                      TRAIN.PRETRAINED_MODEL_DIR pretrained_model/unet_bn_coco \
                      SOLVER.LR 5e-5
```

### 混合精度训练示例

启动混合精度训练，只需将```MODEL.FP16```设置为```True```，具体命令如下
```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch pdseg/train.py --use_gpu \
                      --do_eval \
                      --cfg configs/unet_pet.yaml \
                      BATCH_SIZE 4 \
                      TRAIN.PRETRAINED_MODEL_DIR pretrained_model/unet_bn_coco \
                      SOLVER.LR 5e-5 \
                      MODEL.FP16 True
```
这时候会采用动态scale的方式，若想使用静态scale的方式，可通过```MODEL.SCALE_LOSS```设置，具体命令如下

```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch pdseg/train.py --use_gpu \
                      --do_eval \
                      --cfg configs/unet_pet.yaml \
                      BATCH_SIZE 8 \
                      TRAIN.PRETRAINED_MODEL_DIR pretrained_model/unet_bn_coco \
                      SOLVER.LR 5e-5 \
                      MODEL.FP16 True \
                      MODEL.SCALE_LOSS 512.0
```


### benchmark

| 模型 | 数据集合 | batch size | number gpu cards | 多进程训练 | 混合精度训练 | 速度(image/s) | mIoU on val |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabv3+/Xception65/bn | Cityscapes | 16 | 4 | False | False | 17.27 | 79.20 |
| DeepLabv3+/Xception65/bn | Cityscapes | 16 | 4 | True | False  | 19.80 | 78.90 |
| DeepLabv3+/Xception65/bn | Cityscapes | 16 | 4 | True | True  | 25.84 |79.06|

测试环境：python3.7.3，paddle1.6.0，cuda10，cudnn7.6.2，v100。

### 参考

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
