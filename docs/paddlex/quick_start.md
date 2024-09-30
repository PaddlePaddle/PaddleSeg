# 快速开始

>**说明：**
>* 飞桨一站式全流程开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1)，依托于PaddleSeg的先进技术，支持了图像分割领域的**一站式全流程**开发能力。通过一站式全流程开发，可实现简单且高效的模型使用、组合与定制。
>* PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。本文档提供**图像分割相关产线**的快速使用，单功能模块的快速使用以及更多功能请参考[PaddleSeg一站式全流程开发](./overview.md)中相关章节。


### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的**Python运行环境**。
* **安装PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ 更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗ 更多安装方式参考[PaddleX安装教程](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/installation/installation.md)
### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入图片的本地路径或URL
* `device`: 使用的GPU序号（例如`gpu:0`表示使用第0块GPU），也可选择使用CPU（`cpu`）


以通用语义分割产线为例：
```bash
paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0
```

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/general_object_detection_002.png'}
```
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/03.png)

可视化图片默认保存在 `output` 目录下，您也可以通过 `--save_path` 进行自定义。

其他产线的命令行使用，只需将`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的命令：


| 产线名称      | 使用命令                                                                                                                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用语义分割    | `paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0`                              |
| 图像异常检测    |    `paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0 ` |



### 📝 Python脚本使用

几行代码即可完成产线的快速推理，图像异常检测的Python示例代码如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="anomaly_detection")

output = pipeline.predict("uad_grid.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的`predict` 方法进行推理预测
* 对预测结果进行处理

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/uad_grid.png'}
```
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/02.png)

可视化图片默认保存在 `output` 目录下，您也可以通过 `--save_path` 进行自定义。

其他产线的Python脚本使用，只需将`create_pipeline()`方法的`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：

| 产线名称     | 对应参数                 | 详细说明 |
|----------|----------------------|------|
| 通用语义分割       | `semantic_segmentation` | [通用语义分割产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.md) |
| 图像异常检测       | `anomaly_detection` | [图像异常检测产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.md) |
