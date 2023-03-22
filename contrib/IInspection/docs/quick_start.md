# **Industrial Inspection**
## 简介

旨在创建基于Paddle套件的领先和实用的工业质检行业模型和工具，帮助用户训练更好的模型并将其应用于实践。

## 特性
   * 支持工业质检检测，分割，检测+RoI分割的解决方案的训练和全流程评测；
   * 加入工业质检中的无监督学习算法PactchCore和PaDiM；
   * 支持数据格式转化工具，快速完成检测，分割/RoI分割任务数据格式转化；
   * 支持质检后处理，全流程评测过杀漏失，并进行badcase可视化等；

## 快速开始
### 1. 环境依赖

### 2. 数据准备
   * [准备数据集](./data/prepare_data.md)
   * [数据集格式转换工具](./data/conver_tools.md)
   * [EISeg 数据标注](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/EISeg)

### 3. 训练/推理
   * [无监督异常检测算法](./uad/README.md)
   * [检测分割算法](./det_seg/train_eval.md)

### 全流程预测
   * [准备全流程配置文件](./end2end/parse_config.md)
   * [全流程预测](./end2end/predict.md)

### 全流程评估（过杀/漏检）
   * [指标评估](./end2end/eval.md)
   * [badcase可视化分析](./end2end/eval.md)
   * [后处理参数调优](./end2end/eval.md)
