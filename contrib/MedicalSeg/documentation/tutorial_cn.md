[English](tutorial.md) ｜ 简体中文

这里我们对参数配置、训练、评估、部署等进行了详细的介绍。

## 1. 参数配置
配置文件的结构如下所示：
```bash
├── _base_                   # 一级基础配置，后面所有的二级配置都需要继承它，你可以在这里设置自定义的数据路径，确保它有足够的空间来存储数据。
│   └── global_configs.yml
├── lung_coronavirus         # 每个数据集/器官有个独立的文件夹，这里是 COVID-19 CT scans 数据集的路径。
│   ├── lung_coronavirus.yml # 二级配置，继承一级配置，关于损失、数据、优化器等配置在这里。
│   ├── README.md  
│   └── vnet_lung_coronavirus_128_128_128_15k.yml    # 三级配置，关于模型的配置，不同的模型可以轻松拥有相同的二级配置。
└── schedulers              # 用于规划两阶段的配置，暂时还没有使用它。
    └── two_stage_coarseseg_fineseg.yml
```


## 2. 数据准备
我们使用数据准备脚本来进行一键自动化的数据下载、预处理变换、和数据集切分。只需要运行下面的脚本就可以一键准备好数据：
```
python tools/prepare_lung_coronavirus.py  # 以 CONVID-19 CT scans 为例。
```

## 3. 训练、评估
准备好配置之后，只需要一键运行 [run-vnet.sh](../run-vnet.sh) 就可以进行训练和评估。让我们看看这个脚本中的命令是什么样子的：

```bash
# 设置使用的单卡 GPU id
export CUDA_VISIBLE_DEVICES=0

# 设置配置文件名称和保存路径
yml=vnet_lung_coronavirus_128_128_128_15k
save_dir=saved_model/${yml}
mkdir save_dir

# 训练模型
python3 train.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# 评估模型
python3 val.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams

```


## 4. 模型部署
得到训练好的模型之后，我们可以将它导出为静态图来进行推理加速，下面的步骤就可以进行导出和部署，详细的教程则可以参考[这里](../deploy/python/README.md)：

```bash
cd MedicalSeg/

# 用训练好的模型进行静态图导出
python export.py --config configs/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k.yml --model_path /path/to/your/trained/model

# 使用 Paddle Inference 进行推理
python deploy/python/infer.py \
    --config /path/to/model/deploy.yaml \
    --image_path /path/to/image/path/or/dir/
    --benchmark True   # 在安装了 AutoLog 之后，打开benchmark可以看到推理速度等信息，安装方法可以见 ../deploy/python/README.md

```
如果有“Finish” 输出，说明导出成功，并且可以进行推理加速。

## 5. 在自己的数据上训练
如果你想在自己的数据集上训练，你需要增加一个[数据集代码](../medicalseg/datasets/lung_coronavirus.py), 一个 [数据预处理代码](../tools/prepare_lung_coronavirus.py), 一个和这个数据集相关的[配置目录](../configs/lung_coronavirus), 一份 [训练脚本](../run-vnet.sh)。下面我们分步骤来看这些部分都需要增加什么：

### 5.1 增加配置目录
首先，我们如下图所示，增加一个和你的数据集相关的配置目录：
```
├── _base_
│   └── global_configs.yml
├── lung_coronavirus
│   ├── lung_coronavirus.yml
│   ├── README.md
│   └── vnet_lung_coronavirus_128_128_128_15k.yml
```

### 5.2 增加数据集预处理文件
所有数据需要经过预处理转换成 numpy 数据并进行数据集划分，参考这个[数据预处理代码](../tools/prepare_lung_coronavirus.py)：
```python
├── lung_coronavirus_phase0  # 预处理后的文件路径
│   ├── images
│   │   ├── imagexx.npy
│   │   ├── ...
│   ├── labels
│   │   ├── labelxx.npy
│   │   ├── ...
│   ├── train_list.txt       # 训练数据，格式:  /path/to/img_name_xxx.npy /path/to/label_names_xxx.npy
│   └── val_list.txt         # 评估数据，格式:  img_name_xxx.npy label_names_xxx.npy
```

### 5.3 增加数据集文件
所有的数据集都继承了 MedicalDataset 基类，并通过上一步生成的 train_list.txt 和 val_list.txt 来获取数据。代码示例在[这里](../medicalseg/datasets/lung_coronavirus.py)。

### 5.4 增加训练脚本
训练脚本能自动化训练推理过程，我们提供了一个[训练脚本示例](../run-vnet.sh) 用于参考，只需要复制，并按照需要修改就可以进行一键训练推理：
```bash
# 设置使用的单卡 GPU id
export CUDA_VISIBLE_DEVICES=3

# 设置配置文件名称和保存路径
config_name=vnet_lung_coronavirus_128_128_128_15k
yml=lung_coronavirus/${config_name}
save_dir_all=saved_model
save_dir=saved_model/${config_name}
mkdir -p $save_dir

# 模型训练
python3 train.py --config configs/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# 模型评估
python3 val.py --config configs/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams \

# 模型导出
python export.py --config configs/${yml}.yml \
--model_path $save_dir/best_model/model.pdparams

# 模型预测
python deploy/python/infer.py  --config output/deploy.yaml --image_path data/lung_coronavirus/lung_coronavirus_phase0/images/coronacases_org_007.npy  --benchmark True

```
