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
