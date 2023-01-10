[English](dev_guide_en.md) | 简体中文

# 开发指南

本文档介绍有关本工具箱的高级内容，适合希望为项目贡献代码的开发者阅读。

## 0 索引

+ [1 目录结构](#1-目录结构)
+ [2 关键数据结构](#2-关键数据结构)
+ [3 全景分割标签编码规则](#3-全景分割标签编码规则)
+ [4 自定义数据集开发](#4-自定义数据集开发)

## 1 目录结构

```plain
PanopticSeg
├── configs: 存储配置文件。以模型名称组织各子目录。
├── deploy: 模型部署相关代码。
│   └── python: Python 部署脚本。
├── docs: 文档。
├── paddlepanseg:
│   ├── core: 模型训练、验证与推理核心 API 实现。
│   ├── cvlibs: 核心数据结构相关代码。
│   │   ├── config.py: 用于管理配置项。
│   │   ├── info_dicts.py: 用于存储模型预测结果、样本信息等内容的数据容器。
│   │   └── manager.py: 用于注册和管理组件。
│   ├── datasets: 数据集读取接口的实现。
│   ├── models: 模型具体实现。
│   │   ├── backbones: 骨干网络实现。
│   │   ├── losses: 损失函数实现。
│   │   ├── ops: 外部算子实现。
│   │   └── param_init.py: 用于权重初始化的辅助函数和类。
│   ├── postprocessors: 后处理逻辑实现。
│   ├── transforms: 数据变换（预处理和增强）算子实现。
│   │   └── generate_targets: 训练阶段用于生成损失函数需要的参考输入的算子。
│   └── utils: 辅助函数和类。
├── test_tipc: TIPC 相关。
└── tools: 供用户使用的工具集。
    ├── data: 公开数据集预处理工具。
    ├── analyze_model.py: 用于分析全景分割模型的参数量和计算量。
    ├── export.py: 用于将模型导出为静态图格式。
    ├── predict.py: 用于执行模型推理，并得到可视化结果。
    ├── train.py: 用于模型训练。
    └── val.py: 用于模型精度评估。
```

## 2 关键数据结构

### 2.1 `InfoDict`

`InfoDict` 对象一类特殊的数据容器，可用于存储模型预测结果、预处理的样本信息等内容。`InfoDict` 是本项目中定义的特殊数据结构之一，在 PaddleSeg 中并不存在。`InfoDict` 类的定义位于 `paddlepanseg/cvlibs/info_dicts.py`。

目前总共有四类 `InfoDict`：`SampleDict`，`NetOutDict`，`PPOutDict` 以及 `MetricDict`。这四个类均继承自 `InfoDict` 基类。

+ `SampleDict`：用于存储数据样本的信息（及元信息）。
+ `NetOutDict`：用于存储模型的输出。
+ `PPOutDict`：用于存储后处理器的输出。
+ `MetricDict`：用于存储 `Evaluator` 对象调用 `evaluate()` 方法的输出。

推荐使用工厂函数 `build_info_dict()` 构造 `InfoDict` 对象。该函数定义于 `paddlepanseg/cvlibs/info_dicts.py`。通过指定输入参数 `_type_` 为 `'sample'`、`'net_out'`、`'pp_out'` 或 `'metric'`，可以分别构造以上四种类型的 `InfoDict` 对象。

## 3 全景分割标签编码规则

请参考[此文档](encoding_protocol_cn.md)。

## 4 自定义数据集开发

推荐基于 [MS COCO 格式](https://cocodataset.org/#home)自定义数据集，并实现一个继承自 `paddlepanseg.datasets.base_dataset.COCOStylePanopticDataset` 的数据集接口。具体实现方式如下：

+ 将数据集各类别的元信息存储在一个由字典组成的列表中。可以仿写 `paddlepanseg/datasets/cityscapes.py` 中的 `CITYSCAPES_CATEGORIES`。
+ 定义一个 Python 类，继承自 `paddlepanseg.datasets.base_dataset.COCOStylePanopticDataset`。将类属性 `CATEGORY_META_INFO` 设置为第一步中定义的列表，将类属性 `NUM_CLASSES` 设置为数据集包含的类别数。
+ 重写静态方法 `_get_image_id()`。该方法接受图像路径作为输入，需要返回该图像的唯一标识符。
+ 使用装饰器 `paddlepanseg.cvlibs.manager.DATASETS.add_component()` 修饰写好的类。可参考 `paddlepanseg/datasets/cityscapes.py` 中的做法。