# PaddleSeg 分割库配置说明

PaddleSeg提供了提供了统一的配置用于 训练/评估/可视化/导出模型

配置包含以下Group：

|OPTIONS|用途|支持脚本|
|-|-|-|
|[BASIC](./configs/basic_group.md)|通用配置|ALL|
|[DATASET](./configs/dataset_group.md)|数据集相关|train/eval/vis|
|[MODEL](./configs/model_group.md)|模型相关|ALL|
|[TRAIN](./configs/train_group.md)|训练相关|train|
|[SOLVER](./configs/solver_group.md)|训练优化相关|train|
|[TEST](./configs/test_group.md)|测试模型相关|eval/vis/export_model|
|[AUG](./docs/data_aug.md)|数据增强|ALL|
[FREEZE](./configs/freeze_group.md)|模型导出相关|export_model|
|[DATALOADER](./configs/dataloader_group.md)|数据加载相关|ALL|

`Note`:
 
 代码详见pdseg/utils/config.py
