# 基于Paddle复现
## 1.论文简介
UNETR: Transformers for 3D Medical Image(https://arxiv.org/abs/2103.10504)，

<img src=./documentation/img/unetr_Overview.png></img>

本文介绍了一种新的基于转换器的体系结构，称为UNETR，通过将其重新定义为一维序列序列预测问题，用于体积医学图像的语义分割。
提出使用一个Transformers 编码器来提高模型学习远程依赖关系和在多个尺度上有效捕获全局上下文表示的能力。

**参考实现**：
https://github.com/Project-MONAI/research-contributions/tree/master/UNETR/BTCV
https://github.com/tamasino52/UNETR/blob/main/unetr.py

## 2.复现精度

在msd_brain的测试集的测试效果如下表,达到验收指标，dice-Score=0.718   满足精度要求 0.711

2022-06-02 00:56:06 [INFO]	[EVAL] Class dice: 
[0.9982 0.7936 0.6045 0.7564]


精度和loss可以用visualDL在vdlrecords.1653908268.log`中查看。也可以在train.log中看到训练的详细过程

## 3.环境依赖
通过以下命令安装对应依赖
```shell
!pip install -r PaddleSeg/contrib/MedicalSeg/requirements.txt
```

## 4.数据集

下载地址:

[MSD-Brain Tumor: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ–2)

数据集从下载到解压预处理使用to3d.py处理。执行以下命令。

```shell
python PaddleSeg/contrib/MedicalSeg/tools/to3d.py 1
```

## 5.快速开始

### 模型训练

运行一下命令进行模型训练，在训练过程中会对模型进行评估，启用了VisualDL日志功能，运行之后在`` 文件夹下找到对应的日志文件

```shell
# Train the model: see the train.py for detailed explanation on script args
!python PaddleSeg/contrib/MedicalSeg/train.py --config PaddleSeg/contrib/MedicalSeg/configs/msd_brain/unetr_msd_brain_v100.yml \
--save_dir  saved_model/msd_brain \
--save_interval 300 --log_iters 60 \
--num_workers 4 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  \
>> saved_model/msd_brain/train.log
```

**参数介绍**：
--config:配置路径
--save_dir:模型输出文件夹
--save_interval :保存间隔，单位iter
--log_iters：记录日志间隔，单位iter
--num_workers:读取数据集的worker数量，一般等同于batchsize
--do_eval --use_vdl \   使用eval， 使用vdl记录
--keep_checkpoint_max ：最大保存的检查点数量


其他超参数已经设置好。最后一个epoch结束，模型验证日志在train.log 下面截取一段：
```shell
2022-06-01 04:27:01 [INFO]	[EVAL] The model with the best validation mDice (0.7710) was saved at iter 25500.
2022-06-01 04:30:28 [INFO]	[TRAIN] epoch: 303, iter: 29460/30000, loss: 0.1320, DSC: 87.5789, lr: 0.000003, batch_cost: 3.4591, reader_cost: 2.20177, ips: 1.1564 samples/sec | ETA 00:31:07
2022-06-01 04:33:58 [INFO]	[TRAIN] epoch: 304, iter: 29520/30000, loss: 0.1324, DSC: 87.5916, lr: 0.000002, batch_cost: 3.4920, reader_cost: 2.23871, ips: 1.1455 samples/sec | ETA 00:27:56
2022-06-01 04:37:50 [INFO]	[TRAIN] epoch: 304, iter: 29580/30000, loss: 0.1347, DSC: 87.3389, lr: 0.000002, batch_cost: 3.8649, reader_cost: 2.60926, ips: 1.0349 samples/sec | ETA 00:27:03
2022-06-01 04:41:19 [INFO]	[TRAIN] epoch: 305, iter: 29640/30000, loss: 0.1293, DSC: 87.9317, lr: 0.000002, batch_cost: 3.4876, reader_cost: 2.23526, ips: 1.1469 samples/sec | ETA 00:20:55
2022-06-01 04:45:03 [INFO]	[TRAIN] epoch: 306, iter: 29700/30000, loss: 0.1328, DSC: 87.5170, lr: 0.000002, batch_cost: 3.7331, reader_cost: 2.48128, ips: 1.0715 samples/sec | ETA 00:18:39
2022-06-01 04:45:03 [INFO]	Start evaluating (total_samples: 72, total_iters: 72)...
2022-06-01 04:47:19 [INFO]	[EVAL] #Images: 72, Dice: 0.7671, Loss: 0.238742
2022-06-01 04:47:19 [INFO]	[EVAL] Class dice: 
[0.9979 0.7818 0.5841 0.7045]
2022-06-01 04:47:25 [INFO]	[EVAL] The model with the best validation mDice (0.7710) was saved at iter 25500.
2022-06-01 04:50:45 [INFO]	[TRAIN] epoch: 306, iter: 29760/30000, loss: 0.1314, DSC: 87.7013, lr: 0.000001, batch_cost: 3.3354, reader_cost: 2.07910, ips: 1.1993 samples/sec | ETA 00:13:20
2022-06-01 04:54:24 [INFO]	[TRAIN] epoch: 307, iter: 29820/30000, loss: 0.1304, DSC: 87.7258, lr: 0.000001, batch_cost: 3.6477, reader_cost: 2.39640, ips: 1.0966 samples/sec | ETA 00:10:56
2022-06-01 04:58:04 [INFO]	[TRAIN] epoch: 308, iter: 29880/30000, loss: 0.1257, DSC: 88.1999, lr: 0.000001, batch_cost: 3.6719, reader_cost: 2.41986, ips: 1.0894 samples/sec | ETA 00:07:20
2022-06-01 05:01:45 [INFO]	[TRAIN] epoch: 308, iter: 29940/30000, loss: 0.1307, DSC: 87.7372, lr: 0.000000, batch_cost: 3.6879, reader_cost: 2.43275, ips: 1.0846 samples/sec | ETA 00:03:41
2022-06-01 05:05:12 [INFO]	[TRAIN] epoch: 309, iter: 30000/30000, loss: 0.1318, DSC: 87.6151, lr: 0.000000, batch_cost: 3.4501, reader_cost: 2.19639, ips: 1.1594 samples/sec | ETA 00:00:00
2022-06-01 05:05:12 [INFO]	Start evaluating (total_samples: 72, total_iters: 72)...
2022-06-01 05:07:25 [INFO]	[EVAL] #Images: 72, Dice: 0.7694, Loss: 0.236285
2022-06-01 05:07:25 [INFO]	[EVAL] Class dice: 
[0.998  0.7849 0.5882 0.7066]
2022-06-01 05:07:30 [INFO]	[EVAL] The model with the best validation mDice (0.7710) was saved at iter 25500.
2022-06-01 05:11:10 [INFO]	[TRAIN] epoch: 309, iter: 30060/30000, loss: 0.1299, DSC: 87.8121, lr: 0.000000, batch_cost: 3.6653, reader_cost: 2.40824, ips: 1.0913 samples/sec | ETA 00:00:00
<class 'paddle.nn.layer.conv.Conv3D'>'s flops has been counted
Cannot find suitable count function for <class 'medicalseg.models.unetr_volume_embedding.AbsPositionalEncoding1D'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.common.Dropout'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.fluid.dygraph.container.LayerList'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.common.Linear'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.nn.layer.norm.LayerNorm'>. Treat it as zero FLOPs.
Cannot find suitable count function for <class 'paddle.nn.layer.activation.GELU'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.activation.LeakyReLU'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.nn.layer.norm.InstanceNorm3D'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.conv.Conv3DTranspose'>'s flops has been counted
Total Flops: 179646758912     Total Params: 102108740
```



### 模型验证

除了可以再训练过程中验证模型精度，可以使用val.py脚本进行测试，权重文件可在 链接下载：链接：https://pan.baidu.com/s/1CQF0lI_JZ5sgWuNtzMmU8Q 
提取码：ce7x


```shell
!python PaddleSeg/contrib/MedicalSeg/val.py --config PaddleSeg/contrib/MedicalSeg/configs/msd_brain/unetr_msd_brain_v100.yml --model_path  saved_model/msd_brain//best_model/model.pdparams --save_dir   saved_model/msd_brain/best_model/  --num_workers 1
```
**参数介绍**：


--config:配置路径
- model_path  模型权重所在路径

输出如下：

```shell
  mode: val
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/data/Task01_BrainTumour/Task01_BrainTumour_phase1
  transforms: []
  type: msd_brain_dataset
------------------------------------------------
W0602 00:54:52.900606 28393 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0602 00:54:52.900665 28393 device_context.cc:465] device: 0, cuDNN Version: 7.6.
2022-06-02 00:54:56 [INFO]	Loading pretrained model from saved_model/msd_brain//best_model/model.pdparams
2022-06-02 00:54:59 [INFO]	There are 194/194 variables loaded into UNETR.
2022-06-02 00:54:59 [INFO]	Loaded trained params of model successfully
2022-06-02 00:54:59 [INFO]	Start evaluating (total_samples: 25, total_iters: 25)...
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:253: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.bool, the right dtype will convert to paddle.float32
  format(lhs_dtype, rhs_dtype, lhs_dtype))
25/25 [==============================] - 67s 3s/step - batch_cost: 2.6802 - reader cost: 0.10
2022-06-02 00:56:06 [INFO]	[EVAL] #Images: 25, Dice: 0.7882, Loss: 1.308655
2022-06-02 00:56:06 [INFO]	[EVAL] Class dice: 
[0.9982 0.7936 0.6045 0.7564]
```



### 导出

可以将模型导出，动态图转静态图，使模型预测更快，可以使用export.py脚本进行测试

在这里因为动静态模型转化的问题，修改了stanet的模型代码使其可以转出静态模型。

调试过程中参考这份文档   [报错调试](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/debugging_cn.html)

```shell
!python PaddleSeg/contrib/MedicalSeg/export.py --config PaddleSeg/contrib/MedicalSeg/configs/msd_brain/unetr_msd_brain_v100.yml \
--save_dir  staticmodel/  \
--model_path  saved_model/msd_brain/best_model/model.pdparams \
--without_argmax               \
--input_shape 1 4 128 128 128
```
**参数介绍**：
- input_shape 预测图的形状
-save_dir  静态模型导出路径
- model_path  模型权重所在路径
--without_argmax  不带argmax 


### 使用静态图推理

可以使用unetr_infer.py脚本进行测试

```shell
!python PaddleSeg/contrib/MedicalSeg/deploy/python/unetr_infer.py --config staticmodel/deploy.yaml --image_path miniinfer/images --benchmark True   --batch_size 1
```
**参数介绍**：
--config:导出的静态模型配置路径
- benchmark ：记录推理耗时
- image_path ：用于推理的图片路径
- batch_size 批次

002数据在（0，：，：，100）的原图
<img src=./documentation/img/unetr_raw.png></img>
002数据在（：，：，100）的标签
<img src=./documentation/img/unetr_label.png></img>
002数据在（：，：，100）的预测图
<img src=./documentation/img/unetr_pred.png></img>



### 使用动态图推理

可以使用dynPrediction.py脚本进行测试

```shell
!!python PaddleSeg/contrib/MedicalSeg/dynPrediction.py --config PaddleSeg/contrib/MedicalSeg/configs/msd_brain/unetr_msd_brain_v100.yml --model_path  saved_model/msd_brain/best_model/model.pdparams --image_path miniinfer/images --benchmark True   --batch_size 1
```
**参数介绍**：
--config:动态模型配置路径
--model_path:模型路径
- benchmark ：记录推理耗时
- image_path ：用于推理的图片路径
- batch_size 批次

```shell
W0608 17:10:39.474078 11861 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0608 17:10:39.478483 11861 device_context.cc:465] device: 0, cuDNN Version: 7.6.
2022-06-08 17:10:42 [INFO]	Loading pretrained model from saved_model/msd_brain/best_model/model.pdparams
2022-06-08 17:10:43 [INFO]	There are 194/194 variables loaded into UNETR.
2022-06-08 17:10:43 [INFO]	Loaded trained params of model successfully
2022-06-08 17:10:43 [INFO]	Use GPU
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
2022-06-08 17:11:06 [INFO]	Finish
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:53: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  plat = distro.linux_distribution()[0]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:54: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  ver = distro.linux_distribution()[1]
2022-06-08 17:11:08 [INFO]	

2022-06-08 17:11:08 [INFO]	---------------------- Env info ----------------------
2022-06-08 17:11:08 [INFO]	 OS_version: Ubuntu 16.04
2022-06-08 17:11:08 [INFO]	 CUDA_version: 10.1.243
2022-06-08 17:11:08 [INFO]	 CUDNN_version: 7.3.1
2022-06-08 17:11:08 [INFO]	 drivier_version: 460.32.03
2022-06-08 17:11:08 [INFO]	---------------------- Paddle info ----------------------
2022-06-08 17:11:08 [INFO]	 paddle_version: 2.2.2
2022-06-08 17:11:08 [INFO]	 paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
2022-06-08 17:11:08 [INFO]	 log_api_version: 1.0
2022-06-08 17:11:08 [INFO]	----------------------- Conf info -----------------------
2022-06-08 17:11:08 [INFO]	 runtime_device: None
2022-06-08 17:11:08 [INFO]	 ir_optim: None
2022-06-08 17:11:08 [INFO]	 enable_memory_optim: True
2022-06-08 17:11:08 [INFO]	 enable_tensorrt: None
2022-06-08 17:11:08 [INFO]	 enable_mkldnn: None
2022-06-08 17:11:08 [INFO]	 cpu_math_library_num_threads: None
2022-06-08 17:11:08 [INFO]	----------------------- Model info ----------------------
2022-06-08 17:11:08 [INFO]	 model_name: 
2022-06-08 17:11:08 [INFO]	 precision: fp32
2022-06-08 17:11:08 [INFO]	----------------------- Data info -----------------------
2022-06-08 17:11:08 [INFO]	 batch_size: 1
2022-06-08 17:11:08 [INFO]	 input_shape: dynamic
2022-06-08 17:11:08 [INFO]	 data_num: 4
2022-06-08 17:11:08 [INFO]	----------------------- Perf info -----------------------
2022-06-08 17:11:08 [INFO]	 cpu_rss(MB): 2851.4023, gpu_rss(MB): 8712.0, gpu_util: 84.0%
2022-06-08 17:11:08 [INFO]	 total time spent(s): 19.4287
2022-06-08 17:11:08 [INFO]	 preprocess_time(ms): 391.6706, inference_time(ms): 4465.4858, postprocess_time(ms): 0.0252
```






### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd   AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash PaddleSeg/contrib/MedicalSeg/test_tipc/prepare.sh PaddleSeg/contrib/MedicalSeg/test_tipc/configs/unetr/train_infer_python.txt 'lite_train_lite_infer'

bash PaddleSeg/contrib/MedicalSeg/test_tipc/test_train_inference_python.sh PaddleSeg/contrib/MedicalSeg/test_tipc/configs/unetr/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示

<img src=./documentation/img/unetr_tipc1.png></img>
<img src=./documentation/img/unetr_tipc2.png></img>
<img src=./documentation/img/unetr_tipc3.png></img>
<img src=./documentation/img/unetr_tipc4.png></img>
<img src=./documentation/img/unetr_tipc5.png></img>

## 6.代码结构与详细说明

```
MedicalSeg
├── configs         # All configuration stays here. If you use our model, you only need to change this and run-vnet.sh.
├── data            # Data stays here.
├── test_tipc       # test_tipc stays here.
├── deploy          # deploy related doc and script.
├── medicalseg  
│   ├── core        # the core training, val and test file.
│   ├── datasets  
│   ├── models  
│   ├── transforms  # the online data transforms
│   └── utils       # all kinds of utility files
├── export.py
├── run-vnet.sh     # the script to reproduce our project, including training, validate, infer and deploy
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
├── val.py
└── visualize.ipynb # You can try to visualize the result use this file.

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| UNETR|
|框架版本| PaddlePaddle==2.2.0|


## 8.说明
感谢百度提供的算力，以及举办的本场比赛，让我增强对paddle的熟练度，加深对模型的理解！

## 9.完整的tipc

bash PaddleSeg/contrib/MedicalSeg/test_tipc/prepare.sh
```shell
Archive:  mini_brainT_dataset.zip
   creating: mini_brainT_dataset/
   creating: mini_brainT_dataset/images/
  inflating: mini_brainT_dataset/images/BRATS_002.npy  
  inflating: mini_brainT_dataset/images/BRATS_331.npy  
  inflating: mini_brainT_dataset/images/BRATS_370.npy  
  inflating: mini_brainT_dataset/images/BRATS_390.npy  
   creating: mini_brainT_dataset/labels/
  inflating: mini_brainT_dataset/labels/BRATS_002.npy  
  inflating: mini_brainT_dataset/labels/BRATS_331.npy  
  inflating: mini_brainT_dataset/labels/BRATS_370.npy  
  inflating: mini_brainT_dataset/labels/BRATS_390.npy  
 extracting: mini_brainT_dataset/test_list.txt  
  inflating: mini_brainT_dataset/train_list.txt  
 extracting: mini_brainT_dataset/val_list.txt  
```



bash PaddleSeg/contrib/MedicalSeg/test_tipc/test_train_inference_python.sh
```shell
2022-06-08 17:39:04 [INFO]	
------------Environment Information-------------
platform: Linux-4.15.0-140-generic-x86_64-with-debian-stretch-sid
Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0]
Paddle compiled with cuda: True
NVCC: Cuda compilation tools, release 10.1, V10.1.243
cudnn: 7.6
GPUs used: 1
CUDA_VISIBLE_DEVICES: None
GPU: ['GPU 0: Tesla V100-SXM2-32GB']
GCC: gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
PaddlePaddle: 2.2.2
------------------------------------------------
2022-06-08 17:39:04 [INFO]	
---------------Config Information---------------
batch_size: 2
data_root: PaddleSeg/contrib/MedicalSeg/test_tipc/data
iters: 60
loss:
  coef:
  - 1
  types:
  - coef:
    - 1
    - 1
    losses:
    - type: CrossEntropyLoss
      weight: null
    - type: DiceLoss
    type: MixedLoss
lr_scheduler:
  decay_steps: 60
  end_lr: 0
  learning_rate: 0.0001
  power: 0.9
  type: PolynomialDecay
model:
  dropout: 0.1
  embed_dim: 768
  img_shape: (128, 128, 128)
  input_dim: 4
  num_heads: 12
  output_dim: 4
  patch_size: 16
  type: UNETR
optimizer:
  type: AdamW
  weight_decay: 0.0001
test_dataset:
  dataset_json_path: PaddleSeg/contrib/MedicalSeg/data/Task01_BrainTumour/Task01_BrainTumour_raw/dataset.json
  dataset_root: mini_brainT_dataset
  mode: test
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms: []
  type: msd_brain_dataset
train_dataset:
  dataset_root: mini_brainT_dataset
  mode: train
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms:
  - scale:
    - 0.8
    - 1.2
    size: 128
    type: RandomResizedCrop4D
  - degrees: 90
    rotate_planes:
    - - 1
      - 2
    - - 1
      - 3
    - - 2
      - 3
    type: RandomRotation3D
  - flip_axis:
    - 1
    - 2
    - 3
    type: RandomFlip3D
  type: msd_brain_dataset
val_dataset:
  dataset_json_path: PaddleSeg/contrib/MedicalSeg/data/Task01_BrainTumour/Task01_BrainTumour_raw/dataset.json
  dataset_root: mini_brainT_dataset
  mode: val
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms: []
  type: msd_brain_dataset
------------------------------------------------
W0608 17:39:04.931186 15522 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0608 17:39:04.931231 15522 device_context.cc:465] device: 0, cuDNN Version: 7.6.
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/distributed/parallel.py:136: UserWarning: Currently not a parallel execution environment, `paddle.distributed.init_parallel_env` will not do anything.
  "Currently not a parallel execution environment, `paddle.distributed.init_parallel_env` will not do anything."
nranks数为：
1
0
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:253: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.bool, the right dtype will convert to paddle.float32
  format(lhs_dtype, rhs_dtype, lhs_dtype))
2022-06-08 17:39:25 [INFO]	[TRAIN] epoch: 5, iter: 5/60, loss: 3.8255, DSC: 16.0611, lr: 0.000094, batch_cost: 3.4350, reader_cost: 2.64634, ips: 0.5822 samples/sec | ETA 00:03:08
2022-06-08 17:39:39 [INFO]	[TRAIN] epoch: 10, iter: 10/60, loss: 3.5418, DSC: 17.0561, lr: 0.000086, batch_cost: 2.8494, reader_cost: 2.06981, ips: 0.7019 samples/sec | ETA 00:02:22
2022-06-08 17:39:53 [INFO]	[TRAIN] epoch: 15, iter: 15/60, loss: 3.2770, DSC: 18.1281, lr: 0.000079, batch_cost: 2.7909, reader_cost: 2.01064, ips: 0.7166 samples/sec | ETA 00:02:05
2022-06-08 17:40:07 [INFO]	[TRAIN] epoch: 20, iter: 20/60, loss: 3.1315, DSC: 18.6089, lr: 0.000071, batch_cost: 2.8769, reader_cost: 2.09683, ips: 0.6952 samples/sec | ETA 00:01:55
2022-06-08 17:40:07 [INFO]	Start evaluating (total_samples: 1, total_iters: 1)...
1/1 [==============================] - 5s 5s/step - batch_cost: 5.1366 - reader cost: 2.4462
2022-06-08 17:40:13 [INFO]	[EVAL] #Images: 1, Dice: 0.2031, Loss: 2.766696
2022-06-08 17:40:13 [INFO]	[EVAL] Class dice: 
[0.7464 0.0585 0.0022 0.0055]
2022-06-08 17:40:19 [INFO]	[EVAL] The model with the best validation mDice (0.2031) was saved at iter 20.
2022-06-08 17:40:33 [INFO]	[TRAIN] epoch: 25, iter: 25/60, loss: 2.9820, DSC: 19.2208, lr: 0.000063, batch_cost: 2.8696, reader_cost: 2.08347, ips: 0.6970 samples/sec | ETA 00:01:40
2022-06-08 17:40:48 [INFO]	[TRAIN] epoch: 30, iter: 30/60, loss: 2.8434, DSC: 19.6238, lr: 0.000055, batch_cost: 2.8581, reader_cost: 2.07762, ips: 0.6998 samples/sec | ETA 00:01:25
2022-06-08 17:41:02 [INFO]	[TRAIN] epoch: 35, iter: 35/60, loss: 2.7403, DSC: 19.8057, lr: 0.000047, batch_cost: 2.7920, reader_cost: 2.01129, ips: 0.7163 samples/sec | ETA 00:01:09
2022-06-08 17:41:15 [INFO]	[TRAIN] epoch: 40, iter: 40/60, loss: 2.6681, DSC: 20.0448, lr: 0.000039, batch_cost: 2.7577, reader_cost: 1.97723, ips: 0.7253 samples/sec | ETA 00:00:55
2022-06-08 17:41:15 [INFO]	Start evaluating (total_samples: 1, total_iters: 1)...
1/1 [==============================] - 3s 3s/step - batch_cost: 3.3954 - reader cost: 0.6800
2022-06-08 17:41:19 [INFO]	[EVAL] #Images: 1, Dice: 0.2126, Loss: 2.500392
2022-06-08 17:41:19 [INFO]	[EVAL] Class dice: 
[0.7763 0.0659 0.0018 0.0067]
2022-06-08 17:41:26 [INFO]	[EVAL] The model with the best validation mDice (0.2126) was saved at iter 40.
2022-06-08 17:41:40 [INFO]	[TRAIN] epoch: 45, iter: 45/60, loss: 2.6121, DSC: 20.3129, lr: 0.000030, batch_cost: 2.7992, reader_cost: 2.01426, ips: 0.7145 samples/sec | ETA 00:00:41
2022-06-08 17:41:54 [INFO]	[TRAIN] epoch: 50, iter: 50/60, loss: 2.5624, DSC: 20.5510, lr: 0.000022, batch_cost: 2.8529, reader_cost: 2.07212, ips: 0.7010 samples/sec | ETA 00:00:28
2022-06-08 17:42:08 [INFO]	[TRAIN] epoch: 55, iter: 55/60, loss: 2.5323, DSC: 20.4468, lr: 0.000013, batch_cost: 2.8631, reader_cost: 2.08211, ips: 0.6985 samples/sec | ETA 00:00:14
2022-06-08 17:42:22 [INFO]	[TRAIN] epoch: 60, iter: 60/60, loss: 2.5128, DSC: 20.5044, lr: 0.000003, batch_cost: 2.7651, reader_cost: 1.98544, ips: 0.7233 samples/sec | ETA 00:00:00
2022-06-08 17:42:22 [INFO]	Start evaluating (total_samples: 1, total_iters: 1)...
1/1 [==============================] - 3s 3s/step - batch_cost: 3.3724 - reader cost: 0.6793
2022-06-08 17:42:25 [INFO]	[EVAL] #Images: 1, Dice: 0.2157, Loss: 2.427052
2022-06-08 17:42:25 [INFO]	[EVAL] Class dice: 
[0.7853 0.0685 0.0017 0.0074]
2022-06-08 17:42:40 [INFO]	[EVAL] The model with the best validation mDice (0.2157) was saved at iter 60.
<class 'paddle.nn.layer.conv.Conv3D'>'s flops has been counted
Cannot find suitable count function for <class 'medicalseg.models.unetr_volume_embedding.AbsPositionalEncoding1D'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.common.Dropout'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.fluid.dygraph.container.LayerList'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.common.Linear'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.nn.layer.norm.LayerNorm'>. Treat it as zero FLOPs.
Cannot find suitable count function for <class 'paddle.nn.layer.activation.GELU'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.activation.LeakyReLU'>'s flops has been counted
Cannot find suitable count function for <class 'paddle.nn.layer.norm.InstanceNorm3D'>. Treat it as zero FLOPs.
<class 'paddle.nn.layer.conv.Conv3DTranspose'>'s flops has been counted
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
Total Flops: 179646758912     Total Params: 102108740
!  un successfully with command - python ./PaddleSeg/contrib/MedicalSeg/train.py --config PaddleSeg/contrib/MedicalSeg/test_tipc/configs/unetr/msd_brain_test.yml --save_dir saved_model1/ --save_interval 20 --log_iters 5 --num_workers 2 --do_eval --use_vdl --keep_checkpoint_max 1 --seed 0
2022-06-08 17:42:44 [INFO]	
---------------Config Information---------------
batch_size: 2
data_root: PaddleSeg/contrib/MedicalSeg/test_tipc/data
iters: 60
loss:
  coef:
  - 1
  types:
  - coef:
    - 1
    - 1
    losses:
    - type: CrossEntropyLoss
      weight: null
    - type: DiceLoss
    type: MixedLoss
lr_scheduler:
  decay_steps: 60
  end_lr: 0
  learning_rate: 0.0001
  power: 0.9
  type: PolynomialDecay
model:
  dropout: 0.1
  embed_dim: 768
  img_shape: (128, 128, 128)
  input_dim: 4
  num_heads: 12
  output_dim: 4
  patch_size: 16
  type: UNETR
optimizer:
  type: AdamW
  weight_decay: 0.0001
test_dataset:
  dataset_json_path: PaddleSeg/contrib/MedicalSeg/data/Task01_BrainTumour/Task01_BrainTumour_raw/dataset.json
  dataset_root: mini_brainT_dataset
  mode: test
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms: []
  type: msd_brain_dataset
train_dataset:
  dataset_root: mini_brainT_dataset
  mode: train
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms:
  - scale:
    - 0.8
    - 1.2
    size: 128
    type: RandomResizedCrop4D
  - degrees: 90
    rotate_planes:
    - - 1
      - 2
    - - 1
      - 3
    - - 2
      - 3
    type: RandomRotation3D
  - flip_axis:
    - 1
    - 2
    - 3
    type: RandomFlip3D
  type: msd_brain_dataset
val_dataset:
  dataset_json_path: PaddleSeg/contrib/MedicalSeg/data/Task01_BrainTumour/Task01_BrainTumour_raw/dataset.json
  dataset_root: mini_brainT_dataset
  mode: val
  num_classes: 4
  result_dir: PaddleSeg/contrib/MedicalSeg/test_tipc/data/mini_brainT_dataset
  transforms: []
  type: msd_brain_dataset
------------------------------------------------
W0608 17:42:44.331856 16423 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0608 17:42:44.331907 16423 device_context.cc:465] device: 0, cuDNN Version: 7.6.
2022-06-08 17:42:47 [INFO]	Loading pretrained model from saved_model1/best_model/model.pdparams
2022-06-08 17:42:48 [INFO]	There are 194/194 variables loaded into UNETR.
2022-06-08 17:42:48 [INFO]	Loaded trained params of model successfully
2022-06-08 17:42:48 [INFO]	Start evaluating (total_samples: 1, total_iters: 1)...
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:253: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.bool, the right dtype will convert to paddle.float32
  format(lhs_dtype, rhs_dtype, lhs_dtype))
1/1 [==============================] - 5s 5s/step - batch_cost: 4.5718 - reader cost: 1.9465
2022-06-08 17:42:53 [INFO]	[EVAL] #Images: 1, Dice: 0.2039, Loss: 2.452814
2022-06-08 17:42:53 [INFO]	[EVAL] Class dice: 
[0.7908 0.0217 0.0002 0.0031]
!  un successfully with command - python ./PaddleSeg/contrib/MedicalSeg/val.py --config PaddleSeg/contrib/MedicalSeg/test_tipc/configs/unetr/msd_brain_test.yml --model_path  saved_model1/best_model/model.pdparams --save_dir   saved_model/msd_brain/best_model/  --num_workers 1
W0608 17:42:55.920841 16454 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0608 17:42:55.925145 16454 device_context.cc:465] device: 0, cuDNN Version: 7.6.
2022-06-08 17:42:59 [INFO]	Loaded trained params of model successfully.
[1, 4, 128, 128, 128]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return (isinstance(seq, collections.Sequence) and
2022-06-08 17:43:10 [INFO]	Model is saved in staticmodel1/.
!  un successfully with command - python ./PaddleSeg/contrib/MedicalSeg/export.py --config PaddleSeg/contrib/MedicalSeg/test_tipc/configs/unetr/msd_brain_test.yml --save_dir  staticmodel1/  --model_path  saved_model1/best_model/model.pdparams --without_argmax  --input_shape 1 4 128 128 128
2022-06-08 17:43:13 [INFO]	Use GPU
W0608 17:43:14.633208 16518 analysis_predictor.cc:795] The one-time configuration of analysis predictor failed, which may be due to native predictor called first and its configurations taken effect.
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
I0608 17:43:15.122455 16518 fuse_pass_base.cc:57] ---  detected 48 subgraphs
--- Running IR pass [map_matmul_v2_to_matmul_pass]
I0608 17:43:15.125815 16518 fuse_pass_base.cc:57] ---  detected 24 subgraphs
--- Running IR pass [map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
I0608 17:43:15.143030 16518 fuse_pass_base.cc:57] ---  detected 24 subgraphs
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
--- Running IR pass [conv_elementwise_add_fuse_pass]
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I0608 17:43:15.170677 16518 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0608 17:43:15.454432 16518 memory_optimize_pass.cc:216] Cluster name : tmp_18  size: 1572864
I0608 17:43:15.454478 16518 memory_optimize_pass.cc:216] Cluster name : tmp_9  size: 1572864
I0608 17:43:15.454490 16518 memory_optimize_pass.cc:216] Cluster name : leaky_relu_2.tmp_0  size: 2097152
I0608 17:43:15.454499 16518 memory_optimize_pass.cc:216] Cluster name : conv3d_26.tmp_1  size: 8388608
I0608 17:43:15.454505 16518 memory_optimize_pass.cc:216] Cluster name : instance_norm_11.tmp_2  size: 33554432
I0608 17:43:15.454510 16518 memory_optimize_pass.cc:216] Cluster name : conv3d_31.tmp_1  size: 134217728
I0608 17:43:15.454516 16518 memory_optimize_pass.cc:216] Cluster name : concat_3.tmp_0  size: 268435456
I0608 17:43:15.454520 16518 memory_optimize_pass.cc:216] Cluster name : leaky_relu_1.tmp_0  size: 134217728
I0608 17:43:15.454524 16518 memory_optimize_pass.cc:216] Cluster name : instance_norm_13.tmp_2  size: 134217728
I0608 17:43:15.454528 16518 memory_optimize_pass.cc:216] Cluster name : x  size: 33554432
--- Running analysis [ir_graph_to_program_pass]
I0608 17:43:15.586740 16518 analysis_predictor.cc:714] ======= optimize end =======
I0608 17:43:15.599613 16518 naive_executor.cc:98] ---  skip [feed], feed -> x
I0608 17:43:15.604171 16518 naive_executor.cc:98] ---  skip [conv3d_33.tmp_1], fetch -> fetch
W0608 17:43:16.057713 16518 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0608 17:43:16.061911 16518 device_context.cc:465] device: 0, cuDNN Version: 7.6.
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
2022-06-08 17:43:40 [INFO]	Finish
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:53: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  plat = distro.linux_distribution()[0]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/auto_log/env.py:54: DeprecationWarning: distro.linux_distribution() is deprecated. It should only be used as a compatibility shim with Python's platform.linux_distribution(). Please use distro.id(), distro.version() and distro.name() instead.
  ver = distro.linux_distribution()[1]
2022-06-08 17:43:42 [INFO]	

2022-06-08 17:43:42 [INFO]	---------------------- Env info ----------------------
2022-06-08 17:43:42 [INFO]	 OS_version: Ubuntu 16.04
2022-06-08 17:43:42 [INFO]	 CUDA_version: 10.1.243
2022-06-08 17:43:42 [INFO]	 CUDNN_version: 7.3.1
2022-06-08 17:43:42 [INFO]	 drivier_version: 460.32.03
2022-06-08 17:43:42 [INFO]	---------------------- Paddle info ----------------------
2022-06-08 17:43:42 [INFO]	 paddle_version: 2.2.2
2022-06-08 17:43:42 [INFO]	 paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
2022-06-08 17:43:42 [INFO]	 log_api_version: 1.0
2022-06-08 17:43:42 [INFO]	----------------------- Conf info -----------------------
2022-06-08 17:43:42 [INFO]	 runtime_device: gpu
2022-06-08 17:43:42 [INFO]	 ir_optim: True
2022-06-08 17:43:42 [INFO]	 enable_memory_optim: True
2022-06-08 17:43:42 [INFO]	 enable_tensorrt: False
2022-06-08 17:43:42 [INFO]	 enable_mkldnn: False
2022-06-08 17:43:42 [INFO]	 cpu_math_library_num_threads: 1
2022-06-08 17:43:42 [INFO]	----------------------- Model info ----------------------
2022-06-08 17:43:42 [INFO]	 model_name: 
2022-06-08 17:43:42 [INFO]	 precision: fp32
2022-06-08 17:43:42 [INFO]	----------------------- Data info -----------------------
2022-06-08 17:43:42 [INFO]	 batch_size: 1
2022-06-08 17:43:42 [INFO]	 input_shape: dynamic
2022-06-08 17:43:42 [INFO]	 data_num: 4
2022-06-08 17:43:42 [INFO]	----------------------- Perf info -----------------------
2022-06-08 17:43:42 [INFO]	 cpu_rss(MB): 2980.0508, gpu_rss(MB): 1570.0, gpu_util: 6.0%
2022-06-08 17:43:42 [INFO]	 total time spent(s): 24.1779
2022-06-08 17:43:42 [INFO]	 preprocess_time(ms): 334.418, inference_time(ms): 5710.0455, postprocess_time(ms): 0.0222
```
