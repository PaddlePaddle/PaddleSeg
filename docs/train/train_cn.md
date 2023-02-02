简体中文 | [English](train.md)
# 模型训练

## 1、单卡训练

准备好配置文件后，我们使用`tools/train.py`脚本进行模型训练。

在本文档中我们使用`PP-LiteSeg`模型与`optic_disc`数据集展示训练过程。请确保已经完成了PaddleSeg的安装工作，并且位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # Linux上设置1张可用的卡
# set CUDA_VISIBLE_DEVICES=0  # Windows上设置1张可用的卡

python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

上述训练命令解释：
* `--config`指定配置文件。
* `--save_interval`指定每训练特定轮数后，就进行一次模型保存或者评估（如果开启模型评估）。
* `--do_eval`开启模型评估。具体而言，在训练save_interval指定的轮数后，会进行模型评估。
* `--use_vdl`开启写入VisualDL日志信息，用于VisualDL可视化训练过程。
* `--save_dir`指定模型和visualdl日志文件的保存根路径。

在PP-LiteSeg示例中，训练的模型权重保存在output目录下，如下所示。总共训练1000轮，每500轮评估一次并保存模型信息，所以有`iter_500`和`iter_1000`文件夹。评估精度最高的模型权重，保存在`best_model`文件夹。后续模型的评估、测试和导出，都是使用保存在`best_model`文件夹下精度最高的模型权重。

```
output
  ├── iter_500          #表示在500步保存一次模型
    ├── model.pdparams  #模型参数
    └── model.pdopt     #训练阶段的优化器参数
  ├── iter_1000         #表示在1000步保存一次模型
    ├── model.pdparams  #模型参数
    └── model.pdopt     #训练阶段的优化器参数
  └── best_model        #精度最高的模型权重
    └── model.pdparams  
```

`train.py`脚本输入参数的详细说明如下。

| 参数名              | 用途                                                         | 是否必选项 | 默认值           |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| iters               | 训练迭代次数                                                 | 否         | 配置文件中指定值 |
| batch_size          | 单卡batch size                                               | 否         | 配置文件中指定值 |
| learning_rate       | 初始学习率                                                   | 否         | 配置文件中指定值 |
| config              | 配置文件                                                     | 是         | -                |
| save_dir            | 模型和visualdl日志文件的保存根路径                           | 否         | output           |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据   | 否         | 0                |
| use_vdl             | 是否开启visualdl记录训练数据                                 | 否         | 否               |
| save_interval       | 模型保存的间隔步数                                           | 否         | 1000             |
| do_eval             | 是否在保存模型时启动评估, 启动时将会根据mIoU保存最佳模型至best_model | 否         | 否               |
| log_iters           | 打印日志的间隔步数                                           | 否         | 10               |
| resume_model        | 恢复训练模型路径，如：`output/iter_1000`                     | 否         | None             |
| keep_checkpoint_max | 最新模型保存个数                                             | 否         | 5                |


## 2、多卡训练

使用多卡训练：首先通过环境变量`CUDA_VISIBLE_DEVICES`指定使用的多张显卡，如果不设置`CUDA_VISIBLE_DEVICES`，默认使用所有显卡进行训练；然后使用`paddle.distributed.launch`启动`train.py`脚本进行训练。

多卡训练的`train.py`支持的输入参数和单卡训练相同。

由于Windows环境下不支持nccl，所以无法使用多卡训练。

举例如下，在PaddleSeg根目录下执行如下命令，进行多卡训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡
python -m paddle.distributed.launch tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

## 3、恢复训练：

如果训练中断，我们可以恢复训练，避免从头开始训练。

具体而言，通过给`train.py`脚本设置`resume_model`输入参数，加载中断前最近一次保存的模型信息，恢复训练。

在PP-LiteSeg示例中，总共需要训练1000轮。假如训练到750轮中断了，我们在`output`目录下，可以看到在`iter_500`文件夹中保存了第500轮的训练信息。执行如下命令，加载第500轮的训练信息恢复训练。

```
python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --resume_model output/iter_500 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

单卡和多卡训练都采用相同的方法设置`resume_model`输入参数，即可恢复训练。

## 4、训练可视化

为了直观显示模型的训练过程，对训练过程进行分析从而快速的得到更好的模型，飞桨提供了可视化分析工具：VisualDL。

当`train.py`脚本设置`use_vdl`输入参数后，PaddleSeg会将训练过程中的日志信息写入VisualDL文件，写入的日志信息包括：
* loss
* 学习率lr
* 训练时间
* 数据读取时间
* 验证集上mIoU（当打开了`do_eval`开关后生效）
* 验证集上mean Accuracy（当打开了`do_eval`开关后生效）

在PP-LiteSeg示例中，在训练过程中或者训练结束后，我们都可以通过VisualDL来查看日志信息。

首先执行如下命令，启动VisualDL；然后在浏览器输入提示的网址，效果如下图。

```
visualdl --logdir output/
```

![](./images/fig4.png)
