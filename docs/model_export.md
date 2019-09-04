# 模型导出

通过训练得到一个满足要求的模型后，如果想要将该模型接入到C++预测库或者Serving服务，我们需要通过`pdseg/export_model.py`来导出该模型。

该脚本的使用方法和`train.py/eval.py/vis.py`完全一样

## FLAGS

|FLAG|用途|默认值|备注|
|-|-|-|-|
|--cfg|配置文件路径|None||

## 使用示例

我们使用[训练/评估/可视化](./usage.md)一节中训练得到的模型进行试用，脚本如下

```shell
python pdseg/export_model.py --cfg configs/unet_pet.yaml TEST.TEST_MODEL test/saved_models/unet_pet/final
```

模型会导出到freeze_model目录
