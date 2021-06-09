# 模型导出

本教程提供了一个将训练好的动态图模型转化为静态图模型并进行部署的例子

*注意：如果已经通过量化或者剪枝优化过模型，则模型已经保存为静态图模型，可以直接查看[部署](#模型部署预测)*

## 获取预训练模型

*注意：下述例子为Linux或者Mac上执行的例子，windows请自行在浏览器下载[参数](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams)并存放到所创建的目录*
```shell
mkdir bisenet && cd bisenet
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
cd ..
```

## 将模型导出为静态图模型

请确保完成了PaddleSeg的安装工作，并且位于PaddleSeg目录下，执行以下脚本：

```shell
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --model_path bisenet/model.pdparams
```

### 导出脚本参数解释

|参数名|用途|是否必选项|默认值|
|-|-|-|-|
|config|配置文件|是|-|
|save_dir|模型和visualdl日志文件的保存根路径|否|output|
|model_path|预训练模型参数的路径|否|配置文件中指定值|
|with_softmax|在网络末端添加softmax算子。由于PaddleSeg组网默认返回logits，如果想要部署模型获取概率值，可以置为True|否|False|
|without_argmax|是否不在网络末端添加argmax算子。由于PaddleSeg组网默认返回logits，为部署模型可以直接获取预测结果，我们默认在网络末端添加argmax算子|否|False|

## 结果文件

```shell
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

# 模型部署预测

PaddleSeg目前支持以下部署方式：

|端侧|库|教程|
|-|-|-|
|Python端部署|Paddle预测库|[示例](../deploy/python/)|
|C++端部署|Paddle预测库|[示例](../deploy/cpp/)|
|移动端部署|PaddleLite|[示例](../deploy/lite/)|
|服务端部署|HubServing|完善中|
|前端部署|PaddleJS|[示例](../deploy/web/)|
