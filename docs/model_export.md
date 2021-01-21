# 模型导出



## 获取预训练模型

*注意：下述例子为Linux或者Mac上执行的例子，windows请自行在浏览器下载参数并存放到所创建的目录*
```shell
mkdir bisenet && cd bisenet
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
cd ..
```

## 将模型导出为静态图模型
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

## 结果文件

```shell
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

# 模型部署预测

|端侧|库|教程|
|-|-|-|
|Python端部署|Paddle预测库|[示例](../deploy/python/)|
|移动端部署|ONNX|完善中|
|服务端部署|HubServing|完善中|
|前端部署|PaddleJS|完善中|
