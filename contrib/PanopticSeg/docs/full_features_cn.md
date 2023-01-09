[English](full_features_en.md) | 简体中文

# 完整功能

## 0 索引

+ [1 安装](#1-安装)
+ [2 数据准备](#2-数据准备)
+ [3 配置文件编写](#3-配置文件编写)
+ [4 模型训练](#4-模型训练)
+ [5 模型部署](#5-模型部署)

## 1 安装

### 1.1 安装 PaddlePaddle

+ PaddlePaddle >= 2.4.0
+ Python >= 3.7

由于模型训练需要较高算力，推荐安装 GPU 版本的 PaddlePaddle（支持 CUDA 10.2 或更新的 CUDA 版本）。请访问 [PaddlePaddle 官方网站](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 以阅读详细的安装教程。

### 1.2 下载 PaddleSeg

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

以上操作将默认拉取 PaddleSeg 的最新发行版本。请注意，本工具箱对 PaddleSeg 的版本要求为：

+ PaddleSeg >= 2.8

### 1.3 安装 PaddleSeg 与全景分割工具箱

首先切换到 PaddleSeg 项目的根目录，然后执行如下指令：

```shell
# 安装 PaddleSeg 所需依赖
pip install -r requirements.txt
# 从源码安装 PaddleSeg
pip install .
```

接着，执行如下指令安装本工具箱：

```shell
cd PaddleSeg/contrib/PanopticSeg
# 安装依赖
pip install -r requirements.txt
# 以可编辑模式安装
pip install -e .
```

### 1.4 检查安装情况

执行如下指令：

```shell
python -c "import paddlepanseg; print(paddlepanseg.__version__)"
```

若打印出版本号，则证明安装成功。

## 2 数据准备

本工具箱支持在公开数据集或用户自定义数据集上进行模型的训练和评估。无论对哪种数据集，均要求提供 JSON 格式的全景分割标注文件，以及 PaddleSeg 风格的 *file list*（每个子集都需要）。file list 的编写基本上遵照 [PaddleSeg 规则](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/data/custom/data_prepare_cn.md)，除了使用全景分割参考标签（编码为 RGB 图像）的路径代替语义分割参考标签的路径。

### 2.1 准备公开数据集

对于常用的公开数据集，本工具箱提供数据预处理与 file list 生成的自动化脚本。请参考 `tools/data` 中的内容。

### 2.2 准备自定义数据集

为了在自定义数据集上训练和评估全景分割模型，需要先创建 file list。file list 包含多行，每行表示一个训练或验证样本。在每一行中，都有一对由一个空格分隔的路径，第一条路径指向原始图像，第二条路径指向全景分割标签。例如：

```plain
val2017/000000001000.jpg panoptic_val2017/000000001000.png
val2017/000000001268.jpg panoptic_val2017/000000001268.png
val2017/000000001296.jpg panoptic_val2017/000000001296.png
val2017/000000001353.jpg panoptic_val2017/000000001353.png
val2017/000000001425.jpg panoptic_val2017/000000001425.png
...
```

全景分割标签必须编码为 RGB 图像，每个像素的 R、G、B 通道的像素值按照下式计算：

```plain
r = id % 256
g = id // 256 % 256
b = id // (256 * 256) % 256
```

其中，`id` 是一个标识符，能够唯一标识图像中的一个实例或一个 stuff 区域。

## 3 配置文件编写

类似 PaddleSeg，本工具箱采用模块化的设计方式，定义了数种组件（如数据集、模型等），每种组件都是完全可配置（configurable）的。推荐使用配置文件的方式设置模型训练、评估和部署过程中用于构建组件的超参数。

### 3.1 基本规则

由于本工具箱基于 PaddleSeg API 实现，因此配置文件的基本编写规则与 PaddleSeg 一致。详细内容请参考 [PaddleSeg 文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/config/pre_config_cn.md)。

### 3.2 后处理器

除了基本规则，本工具箱还根据全景分割任务的特性增加了一些可配置的新组件，如后处理器（postprocessor）。后处理器主要用于将网络的输出转换为最终的全景分割结果（包含图像中所有实例的 ID 和每个像素的语义类别）。

可通过新增或修改配置文件中 `postprocessor` 键值对的方式配置用于构造后处理器的参数：

```yaml
postprocessor:
  type: MaskFormerPostprocessor
  num_classes: 19
  object_mask_threshold: 0.8
  overlap_threshold: 0.8
  label_divisor: 1000
  ignore_index: 255
```

所有的后处理器都支持配置 `num_classes`、`thing_ids`、`label_divisor` 和 `ignore_index`。如果以上四个配置项中的某项未在配置文件中被指定，将首先尝试从 `val_dataset` 中解析相关配置项，如果未找到，则使用预设的默认值。

### 3.3 `Collect`

与 PaddleSeg 不同，本工具箱的**所有**数据变换（data transformation）配置均需以 `Collect` 算子结尾。这是由于本工具箱的数据预处理基于 [`InfoDict`](dev_guide_cn.md#21-infodict)，而 `Collect` 算子被用于从 `InfoDict` 对象中提取需要用到的键值对。

在训练阶段，一般需要提取如下键值对：

+ `img`: 经过预处理的模型输入。
+ `label`: 经过预处理的参考标签。
+ `img_path`: 原始图像路径。
+ `lab_path`: 原始参考标签路径。
+ `img_h`: 原始图像高度。
+ `img_w`: 原始图像宽度。

在评估阶段，一般需要提取如下键值对：

+ `img`: 经过预处理的模型输入。
+ `label`: 经过预处理的参考标签。
+ `ann`: 从 JSON 格式标注文件中解析的标注信息。
+ `image_id`: 原始图像 ID。
+ `gt_fields`: 保存有一系列键，指定 `InfoDict` 中的哪些项应该被标记为参考标签。
+ `trans_info`: 保存有一系列记录张量形状的元信息，用于恢复图像尺寸。
+ `img_path`: 原始图像路径。
+ `lab_path`: 原始参考标签路径。
+ `pan_label`: 经过预处理的全景分割参考标签。
+ `sem_label`: 经过预处理的语义分割参考标签。
+ `ins_label`: 经过预处理的实例分割参考标签。

## 4 模型训练

### 4.1 进行模型训练

训练基础指令如下：

```shell
python tools/train.py \
    --config {配置文件路径}
```

通过命令行选项可以复写部分配置项，这在需要在某次训练临时修改一小部分超参数时很有用。`tools/train.py` 所支持的命令行选项与 PaddleSeg 的训练脚本十分接近。请参考 [PaddleSeg 文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/train/train_cn.md) 并执行如下指令以获取完整的命令行选项列表：

```shell
python tools/train.py --help
```

部分较为重要的命令行选项如下：

+ `--config`: 配置文件路径。
+ `--save_dir`: 保存模型训练过程中检查点（checkpoint）的路径。
+ `--num_workers`: 数据预加载（data prefetching）使用的进程数量。
+ `--do_eval`: 指定此选项以在训练过程中周期性评估模型精度（执行模型验证）。
+ `--log_iters`: 打印日志的间隔（单位为迭代数）。
+ `--eval_sem`: 指定此选项以在模型验证阶段计算语义分割精度指标（如 mIoU）。
+ `--eval_ins`: 指定此选项以在模型验证阶段计算实例分割精度指标（如 mAP）。
+ `--debug`: 启用调试模式。在调试模式下，当程序因为异常而终止时，将在异常发生位置自动添加一个 [pdb](https://docs.python.org/3/library/pdb.html) 断点以便进行事后调试（post-mortem debugging）。

如果需要将日志信息存储到文件中，可以考虑如下方式：

```shell
TAG='mask2former'
python tools/train.py \
    --config configs/mask2former/mask2former_resnet50_os16_coco_1024x1024_bs4_370k.yml \
    --log_iters 50 \
    --num_workers 4 \
    --do_eval \
    --eval_sem \
    --eval_ins \
    --save_dir "output/${TAG}" \
    2>&1 \
    | tee "output/train_${TAG}.log"
```

如果需要使用多块 GPU 进行训练，执行如下指令：

```shell
# 指定要使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch tools/train.py \
    --config {配置文件路径}
```

**请注意，某些模型可能包含『前置条件』**，例如，在执行模型训练、评估等步骤前可能需要编译外部 C++/CUDA 算子。请在 `config` 的相关文档中阅读更多细节。

### 4.2 进行模型精度评估

在训练过程中或训练完成后，在训练脚本的 `--save_dir` 选项指定的目录（默认为 `output`）中将存储模型权重等训练结果。可通过如下指令对验证集上 [PQ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) 指标最高的模型权重（也就是 `output/best_model`）进行精度评估：

```shell
python tools/val.py \
    --config {配置文件路径} \
    --model_path output/best_model/model.pdparams
```

可通过修改 `--model_path` 选项对其它模型权重进行精度评估。执行 `python tools/val.py --help` 可以查看脚本支持的完整命令行选项列表。

与 `tools/train.py` 一样，在精度评估时，指定 `--eval_sem` 与 `--eval_ins` 将分别计算语义分割与实例分割的指标，但这可能会延长精度评估的耗时。

### 4.3 获取可视化结果

执行如下指令以获取单张或多张图像的预测结果：

```shell
python tools/predict.py \
    --config {配置文件路径} \
    --model_path {模型路径} \
    --image_path {单幅图像路径或包含图像的目录} \
    --save_dir {结果保存路径}
```

执行 `python tools/predict.py --help` 可查看完整的命令行参数列表。

执行完上述脚本后，可在 `--save_dir` 指定的路径中获取可视化结果。对于每幅图像，可视化结果有三种：

+ `{图像名称前缀}_sem.png`：用不同颜色表示不同的语义类别。

<img src="https://user-images.githubusercontent.com/21275753/210925337-797befea-b774-4d63-849b-574709f098c7.png" height="300">

+ `{图像名称前缀}_ins.png`：用不同颜色表示不同的实例。对于 stuff 类别，该类别的所有像素被当作属于**一个**实例。

<img src="https://user-images.githubusercontent.com/21275753/210925345-773f7c81-d281-4053-9684-6e8e6ac841f9.png" height="300">

+ `{图像名称前缀}_pan.png`：用不同的*基准*颜色表示不同的语义类别。在此基础上，为每个实例添加颜色偏移以区分不同实例。

<img src="https://user-images.githubusercontent.com/21275753/210925355-262775c2-3a9d-4c31-b45a-cef3bdebf4e0.png" height="300">

## 5 模型部署

### 5.1 导出模型

为了获得更高的推理效率，推荐在模型部署时将模型转换为静态图格式。执行如下命令：

```shell
python tools/export.py \
    --config {配置文件路径} \
    --model_path {模型路径} \
    --save_dir {导出模型保存路径} \
    --input_shape {输入张量形状}
```

请参考 [PaddleSeg 文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/model_export_cn.md)以获取关于模型导出的更多信息。请注意，部分全景分割模型目前尚不支持导出为静态图格式。请参考 `configs` 中的相关文档以确认模型是否支持导出。

### 5.2 使用 Paddle-Inference API 执行预测

执行如下指令，基于 [Paddle-Inference](https://paddle-inference.readthedocs.io/en/master/) API 进行预测：

```shell
python deploy/python/infer.py \
    --config {部署配置文件路径} \
    --image_path {单幅图像路径或包含图像的目录}
```

请注意，`{部署配置文件路径}` 是导出模型的目录中 `deploy.yaml` 文件的路径。

输出的预测结果是一幅 RGB 图像，其各通道像素值为：

```plain
r = pan_id % 256
g = pan_id // 256 % 256
b = pan_id // (256 * 256) % 256
```

其中，`pan_id` 由后处理器计算得到，标识图像中的一个实例或一个 stuff 区域。关于 `pan_id` 的更详细说明，请参考[此文档](encoding_protocol_cn.md)。
