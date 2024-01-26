English | [简体中文](distributed_train_cn.md)

## 1. Distributed Training


Distributed training refers to splitting the training task into multiple computing nodes for calculation according to a certain method, then aggregating and updating the gradient and other information calculated after splitting using some methods. Distributed training is able to accelerate the training speed.

Paddle distributed training technology is derived from Baidu's business practice and has been applied in the fields of natural language processing, computer vision, search and recommendation through large-scale businesses. High performance of distributed training is one of the core advantages of PaddlePaddle. 

PaddleSeg supports both single machine training and multi machine training. For more methods and documents about distributed training, please refer to: [Quick Start Tutorial for Distributed Training](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。



## 2. Usage

### 2.1 Single-machine

Take PP-LiteSeg as an example, after preparing the data locally, use the interface of `paddle.distributed.launch` or `fleetrun` to start the training task. Below is an example of running the script.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k
python -m paddle.distributed.launch \
    --log_dir=./log/ \
    --gpus "0,1,2,3,4,5,6,7" \
    train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 1000 \
    --num_workers 3 \
    --do_eval \
    --use_vdl
```

### 2.2 Multi-machine

Compared with single-machine training, when training on multiple machines, you only need to add the `--ips` parameter, which indicates the ip list of machines that need to participate in distributed training. The ips of different machines are separated by commas. Below is an example of running code.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k
ip_list="a1.b1.c1.d1,a2.b2.c2.d2"
python -m paddle.distributed.launch \
    --log_dir=./log/ \
    --ips=${ip_list} \
    --gpus "0,1,2,3,4,5,6,7" \
    train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 1000 \
    --num_workers 3 \
    --do_eval \
    --use_vdl
```

**Note:**
* The ip information of different machines needs to be separated by commas, which can be viewed through `ifconfig` or `ipconfig`.
* Password-free settings are required between different machines, and they can be pinged directly, otherwise the communication cannot be completed.
* The code, data, and running commands or scripts between different machines need to be consistent, and the set training commands or scripts need to be run on all machines. The first device of the first machine in the final `ip_list` is trainer0, and so on.
* The starting port of different machines may be different. It is recommended to set the same starting port for multi-machine running in different machines before starting the multi-machine task. The command is `export FLAGS_START_PORT=17000`, and the port value is recommended to be `10000~20000`.


## 3. Performance

We conducted model training on 3x8 V100 GPUs. Accuracy, training time, and multi machine acceleration ratio of different models are shown below.

| Model    | Backbone | Dataset | Configuration   | 8 GPU training time / Accuracy | 3x8 GPU training time / Accuracy | Acceleration ratio  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  OCRNet | HRNet_w18 | Cityscapes | [ocrnet_hrnetw18_cityscapes_1024x512_160k.yml](../../configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml)  | 8.9h/80.91% | 5.33h/80.13%  | **1.88** |
|  SegFormer_B0 | - | Cityscapes | [segformer_b0_cityscapes_1024x1024_160k.yml](../../configs/segformer/segformer_b0_cityscapes_1024x1024_160k.yml)  | 5.61h/76.73% | 2.6h/75.86%  | **2.15** |
|  SegFormer_B0<sup>*</sup> | - | Cityscapes | [segformer_b0_cityscapes_1024x1024_160k.yml](../../configs/segformer/segformer_b0_cityscapes_1024x1024_160k.yml)  | 5.61h/76.73% | 3.5h/76.48%  | **1.60** |


We conducted model training on 4x8 V100 GPUs. Accuracy, training time, and multi machine acceleration ratio of different models are shown below.


| Model    | Backbone | Dataset | Configuration   | 8 GPU training time / Accuracy | 4x8 GPU training time / Accuracy | Acceleration ratio  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  PP-LiteSeg-T  | STDC1 | Cityscapes | [pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml](../../configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml)  | 7.58h/73.05% | 2.5h/72.43%%  | **3.03** |


**Note**
* When the number of GPU cards for training is too large, the accuracy will be slightly lost (about 1%). At this time, you can try to warmup the training process or increase some training epochs to reduce the lost.
* SegFormer_B0<sup>*</sup> means increasing training iterations of SegFormer_B0 ( Defaulf as 26.7k for 3x8 GPUs, here it is set as 35k).
