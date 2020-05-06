# 遥感分割（RemoteSensing）
遥感影像分割是图像分割领域中的重要应用场景，广泛应用于土地测绘、环境监测、城市建设等领域。遥感影像分割的目标多种多样，有诸如积雪、农作物、道路、建筑、水源等地物目标，也有例如云层的空中目标。

PaddleSeg提供了针对遥感专题的语义分割库，涵盖图像预处理、数据增强、模型训练、预测流程，帮助大家利用深度学习技术解决遥感影像分割问题。

**Note:** 所有命令需要在`PaddleSeg/contrib/RemoteSensing/`目录下执行。

## 前置依赖
- Paddle 1.7.1+
- Python 3.0+
- Numpy
- PIL
- OpenCV

PaddlePaddle的安装, 请按照[官网指引](https://paddlepaddle.org.cn/install/quick)安装合适自己的版本。

## 目录结构说明
 ```
RemoteSensing               # 根目录
 |-- models                 # 模型类定义模块
 |-- nets                   # 组网模块
 |-- readers                # 数据读取模块
 |-- transforms             # 数据增强模块
 |-- utils                  # 公用模块
 |-- main.py                # 主函数（使用demo）
 |-- README.md              # 使用手册

 ```
## 数据协议
由于遥感领域图像格式多种多样，不同传感器产生的数据格式可能不同。本分割库目前采用npy格式作为遥感数据的格式，采用png无损压缩格式作为标注图片格式。

**标注协议** 采用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别。标注类别最多为256类。

### 文件列表

本分割库采用通用的文件列表方式组织训练集、验证集和测试集。在训练、预测过程前必须准备好相应的文件列表。

文件列表组织形式如下
```
原始图片路径 [SEP] 标注图片路径
```

其中`[SEP]`是文件路径分割符，需为空格。文件列表的路径以数据集根目录作为相对路径起始点。具体要求和如何生成文件列表参考[文件列表规范](../../docs/data_prepare.md#文件列表)。

## 数据预处理
您需要将自己的数据转换为本分割库的格式，可分别使用numpy和pil库保存遥感数据和标注图片。其中numpy api示例如下：
```python
import numpy as np

# 保存遥感数据
# img类型：numpy.ndarray
np.save(save_path, img)
```


## 使用方式

main.py包含模型训练和预测的示例代码，按照您的机器环境和数据修改如下配置。

```python
# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# batch size
batch_size = 4
# training epochs
epochs = 100

# number of data channel
channel = 10
# model save directory
save_dir = ''
# dataset directory
data_dir = ''
```

执行main.py即可快速运行。
```shell script
python main.py
````

也可以使用`RemoteSensing`目录下提供的api构建您自己的分割代码。
