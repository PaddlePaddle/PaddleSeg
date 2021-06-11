# 模型压缩教程

许多的神经网络模型需要巨大的计算开销和内存开销，严重阻碍了资源受限下的使用，通过模型压缩可以减少模型参数或者计算量，有效地降低计算和存储开销，便于部署再受限的硬件环境中。PaddleSeg基于PaddleSlim，集成了模型在线量化、模型裁剪等模型压缩能力。本文档提供相关能力的使用教程。

## 安装PaddleSlim

在进行模型量化或者裁剪前，请先安装相关依赖：

```shell
pip install paddleslim==2.0.0
```

## 模型量化

模型量化是通过将模型中的参数计算，从浮点计算转成低比特定点计算的技术，可以有效的降低模型计算强度、参数大小和内存消耗。PaddleSeg基于PaddleSlim库，提供了模型在线量化的能力。

*注意：对于小模型而言，由于模型本身运行速度已经非常快，加入量化操作反而可能导致模型运行速度变慢*

### step 1. 模型训练

我们可以通过PaddleSeg提供的脚本对模型进行训练，请确保完成了PaddleSeg的安装工作，并且位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### step 2. 模型量化并再次训练

加载上一步训练完成的模型，启动量化脚本，并配置对应的参数。

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|retraining_iters|量化完成后的重训练迭代数|是||
|config|配置文件|是||
|batch_size|重训练时的单卡batch size|否|配置文件中指定值|
|learning_rate|重训练时的学习率|否|配置文件中指定值|
|model_path|预训练模型参数路径|否||
|num_workers|重训练时用于异步读取数据的进程数量，大于等于1时开启子进程读取数据|否|0|
|save_dir|量化后模型的保存路径|否|output|

```shell
# 请在PaddleSeg根目录运行
export PYTHONPATH=`pwd`
# windows下请执行以下命令
# set PYTHONPATH=%cd%

python slim/quant.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --retraining_iters 10 \
       --model_path output/best_model/model.pdparams \
       --save_dir quant_model
```

## 模型裁剪

模型裁剪，是指通过减少卷积层中卷积核的数量，来减小模型大小和降低模型计算复杂度的一种模型压缩方式。PaddleSeg基于PaddleSlim库，提供了基于敏感度的卷积通道剪裁脚本，能够快速地分析出模型中的冗余参数，按照用户指定的裁剪比例进行剪枝并重新训练，在精度和速度上取得一个较好的平衡。

*注意：目前只有以下模型支持裁剪功能，更多模型正在支持中：*
*BiSeNetv2、FCN、Fast-SCNN、HardNet、UNet*

### step 1. 模型训练

我们可以通过PaddleSeg提供的脚本对模型进行训练，请确保完成了PaddleSeg的安装工作，并且位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### step 2. 裁剪并保存模型

加载上一步训练完成的模型，指定裁剪率，并启动裁剪脚本。

*注意：基于敏感度的卷积通道剪裁方式，需要不断的评估每个卷积核对于最终精度的影响，因此耗时会比较久*

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|pruning_ratio|卷积核裁剪比率|是||
|retraining_iters|裁剪完成后的重训练迭代数|是||
|config|配置文件|是||
|batch_size|重训练时的单卡batch size|否|配置文件中指定值|
|learning_rate|重训练时的学习率|否|配置文件中指定值|
|model_path|预训练模型参数路径|否||
|num_workers|重训练时用于异步读取数据的进程数量，大于等于1时开启子进程读取数据|否|0|
|save_dir|裁剪后模型的保存路径|否|output|

```shell
# 请在PaddleSeg根目录运行
export PYTHONPATH=`pwd`
# windows下请执行以下命令
# set PYTHONPATH=%cd%

python slim/prune.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --pruning_ratio 0.2 \
       --model_path output/best_model/model.pdparams \
       --retraining_iters 100 \
       --save_dir prune_model
```

## 部署

通过`量化`和`剪枝`得到的模型，我们可以直接进行部署应用，相关教程请参考**模型部署**

## 量化&剪枝加速比

    测试环境：
       GPU: V100
       CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
       CUDA: 10.2
       cuDNN: 7.6
       TensorRT: 6.0.1.5

    测试方法:
       1. 运行耗时为纯模型预测时间，测试图片cityspcaes(1024x2048)
       2. 预测10次作为热启动，连续预测50次取平均得到预测时间
       3. 使用GPU + TensorRT测试

|模型|未量化运行耗时(ms)|量化运行耗时(ms)|加速比|
|-|-|-|-|
|deeplabv3_resnet50_os8|204.2|150.1|26.49%|
|deeplabv3p_resnet50_os8|147.2|89.5|39.20%|
|gcnet_resnet50_os8|201.8|126.1|37.51%|
|pspnet_resnet50_os8|266.8|206.8|22.49%|

|模型|裁剪率|运行耗时(ms)|加速比|
|-|-|-|-|
|fastscnn|-|7.0|-|
||0.1|5.9|15.71%|
||0.2|5.7|18.57%|
||0.3|5.6|20.00%|
|fcn_hrnetw18|-|43.28|-|
||0.1|40.46|6.51%|
||0.2|40.41|6.63%|
||0.3|38.84|10.25%|
|unet|-|76.04|-|
||0.1|74.39|2.16%|
||0.2|72.10|5.18%|
||0.3|66.96|11.94%|
