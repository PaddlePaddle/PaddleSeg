简体中文 | [English](model_export_onnx.md)
# 导出ONNX格式模型

PaddleSeg训练好模型后，也支持导出ONNX格式模型，本教程提供一个示例介绍使用方法。

导出ONNX格式模型的完整方法，请参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

## 1. 导出预测模型

参考[文档](./model_export.md)导出预测模型。

复用[文档](./model_export.md)中的示例，成功将导出的预测模型文件保存在output文件夹中，如下。

```shell
./output
  ├── deploy.yaml            # 部署相关的配置文件，主要说明数据预处理的方式
  ├── model.pdmodel          # 预测模型的拓扑结构文件
  ├── model.pdiparams        # 预测模型的权重文件
  └── model.pdiparams.info   # 参数额外信息，一般无需关注
```

## 2. 导出ONNX格式模型

安装Paddle2ONNX（高于或等于0.6版本)。

```
pip install paddle2onnx
```

执行如下命令，使用Paddle2ONNX将output文件夹中的预测模型导出为ONNX格式模型。

```
paddle2onnx --model_dir output \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file output.onnx
```

导出的ONNX格式模型保存为output.onnx文件。

参考文档：
* [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
* [ONNX](https://onnx.ai/)
