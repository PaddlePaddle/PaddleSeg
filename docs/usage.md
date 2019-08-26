PaddleSeg提供了 `训练`/`评估`/`预测(可视化)`/`模型导出` 等四个功能的使用脚本。四个脚本都支持通过不同的Flags来开启特定功能，也支持通过Options来修改默认的[训练配置](./config.md)。四者的使用方式非常接近，如下：

```shell
# 训练
python pdseg/train.py ${FLAGS} ${OPTIONS}
# 评估
python pdseg/eval.py ${FLAGS} ${OPTIONS}
# 预测/可视化
python pdseg/vis.py ${FLAGS} ${OPTIONS}
# 模型导出
python pdseg/export_model.py ${FLAGS} ${OPTIONS}
```

`Note`:

> * FLAGS必须位于OPTIONS之前，否会将会遇到报错，例如如下的例子:
>
> ```shell
> # FLAGS "--cfg configs/cityscapes.yaml" 必须在 OPTIONS "BATCH_SIZE 1" 之前
> python pdseg/train.py BATCH_SIZE 1 --cfg configs/cityscapes.yaml
> ```

## FLAGS

|FLAG|支持脚本|用途|默认值|备注|
|-|-|-|-|-|
|--cfg|ALL|配置文件路径|None||
|--use_gpu|train/eval/vis|是否使用GPU进行训练|False||
|--use_mpio|train/eval|是否使用多线程进行IO处理|False|打开该开关会占用一定量的CPU内存，但是可以提高训练速度。</br> NOTE：windows平台下不支持该功能, 建议使用自定义数据初次训练时不打开，打开会导致数据读取异常不可见。 </br> |
|--use_tb|train|是否使用TensorBoard记录训练数据|False||
|--log_steps|train|训练日志的打印周期（单位为step）|10||
|--debug|train|是否打印debug信息|False|IOU等指标涉及到混淆矩阵的计算，会降低训练速度|
|--tb_log_dir|train|TensorBoard的日志路径|None||
|--do_eval|train|是否在保存模型时进行效果评估|False||
|--vis_dir|vis|保存可视化图片的路径|"visual"||
|--also_save_raw_results|vis|是否保存原始的预测图片|False||

## OPTIONS

详见[训练配置](./config.md)

## 使用示例
下面通过一个简单的示例，说明如何使用PaddleSeg提供的预训练模型进行finetune。我们选择基于COCO数据集预训练的unet模型作为pretrained模型，在一个Oxford-IIIT Pet数据集上进行finetune。
**Note:** 为了快速体验，我们使用Oxford-IIIT Pet做了一个小型数据集，后续数据都使用该小型数据集。

### 准备工作
在开始教程前，请先确认准备工作已经完成：
1. 下载合适版本的paddlepaddle
2. PaddleSeg相关依赖已经安装

如果有不确认的地方，请参考[安装说明](./docs/installation.md)

### 下载预训练模型
```shell
# 下载预训练模型
wget https://bj.bcebos.com/v1/paddleseg/models/unet_coco_init.tgz
# 解压缩到当前路径下
tar xvzf unet_coco_init.tgz
```
### 下载Oxford-IIIT数据集
```shell
# 下载Oxford-IIIT Pet数据集
wget https://paddleseg.bj.bcebos.com/dataset/mini_pet.zip --no-check-certificate
# 解压缩到当前路径下
unzip mini_pet.zip
```

### Finetune
接着开始Finetune，为了方便体验，我们在configs目录下放置了Oxford-IIIT Pet所对应的配置文件`unet_pet.yaml`，可以通过`--cfg`指向该文件来设置训练配置。

我们选择两张GPU进行训练，这可以通过环境变量`CUDA_VISIBLE_DEVICES`来指定。

除此之外，我们指定总BATCH_SIZE为4，PaddleSeg会根据可用的GPU数量，将数据平分到每张卡上，务必确保BATCH_SIZE为GPU数量的整数倍（在本例中，每张卡的BATCH_SIZE为2）。

```
export CUDA_VISIBLE_DEVICES=0,1
python pdseg/train.py --use_gpu \
                      --do_eval \
                      --use_tb \
                      --tb_log_dir train_log \
                      --cfg configs/unet_pet.yaml \
                      BATCH_SIZE 4 \
                      TRAIN.PRETRAINED_MODEL unet_coco_init \
                      DATASET.DATA_DIR mini_pet \
                      DATASET.TEST_FILE_LIST mini_pet/file_list/test_list.txt \
                      DATASET.TRAIN_FILE_LIST mini_pet/file_list/train_list.txt \
                      DATASET.VAL_FILE_LIST mini_pet/file_list/val_list.txt \
                      DATASET.VIS_FILE_LIST mini_pet/file_list/val_list.txt \
                      TRAIN.SYNC_BATCH_NORM True \
                      SOLVER.LR 5e-5
```

`NOTE`:

> * 上述示例中，一共存在三套配置方案: PaddleSeg默认配置/unet_pet.yaml/OPTIONS，三者的优先级顺序为 OPTIONS > yaml > 默认配置。这个原则对于train.py/eval.py/vis.py/export_model.py都适用
>
> * 如果发现因为内存不足而Crash。请适当调低BATCH_SIZE。如果本机GPU内存充足，则可以调高BATCH_SIZE的大小以获得更快的训练速度
>
> * windows并不支持多卡训练

### 训练过程可视化

当打开do_eval和use_tb两个开关后，我们可以通过TensorBoard查看训练的效果
```shell
tensorboard --logdir train_log --host {$HOST_IP} --port {$PORT}
```

NOTE:
1. 上述示例中，$HOST_IP为机器IP地址，请替换为实际IP，$PORT请替换为可访问的端口
2. 数据量较大时，前端加载速度会比较慢，请耐心等待

启动TensorBoard命令后，我们可以在浏览器中查看对应的训练数据
在`SCALAR`这个tab中，查看训练loss、iou、acc的变化趋势
![](docs/imgs/tensorboard_scalar.JPG)

在`IMAGE`这个tab中，查看样本的预测情况
![](docs/imgs/tensorboard_image.JPG)

### 模型评估
训练完成后，我们可以通过eval.py来评估模型效果。由于我们设置的训练EPOCH数量为500，保存间隔为10，因此一共会产生50个定期保存的模型，加上最终保存的final模型，一共有51个模型。我们选择最后保存的模型进行效果的评估：
```shell
python pdseg/eval.py --use_gpu \
                     --cfg configs/unet_pet.yaml \
                     DATASET.DATA_DIR mini_pet \
                     DATASET.VAL_FILE_LIST mini_pet/file_list/val_list.txt \
                     TEST.TEST_MODEL test/saved_models/unet_pet/final
```


### 模型预测/可视化
通过vis.py来评估模型效果，我们选择最后保存的模型进行效果的评估：
```shell
python pdseg/vis.py --use_gpu \
                     --cfg configs/unet_pet.yaml \
                     DATASET.DATA_DIR mini_pet \
                     DATASET.TEST_FILE_LIST mini_pet/file_list/test_list.txt \
                     TEST.TEST_MODEL test/saved_models/unet_pet/final
```
`NOTE`
1. 可视化的图片会默认保存在visual/visual_results目录下，可以通过`--vis_dir`来指定输出目录
2. 训练过程中会使用DATASET.VIS_FILE_LIST中的图片进行可视化显示，而vis.py则会使用DATASET.TEST_FILE_LIST

### 模型导出
当确定模型效果满足预期后，我们需要通过export_model.py来导出一个可用于部署到服务端预测的模型：
```shell
python pdseg/export_model.py --cfg configs/unet_pet.yaml \
                                   TEST.TEST_MODEL test/saved_models/unet_pet/final
```

模型会导出到freeze_model目录，接下来就是进行模型的部署，相关步骤，请查看[模型部署](./inference/README.md)
