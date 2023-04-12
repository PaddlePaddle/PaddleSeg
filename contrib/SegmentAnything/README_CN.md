# Segment Anything with PaddleSeg

## 参考文献

> Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick. [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/).

## 目录

1. 概述
2. 性能展示
3. 一行代码体验

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> 概述

我们使用PaddlePaddle框架实现了“Segment Anything Model”（SAM），这是一项新的任务、模型和图像分割数据集。SAM可以根据不同类型的提示（包括点、框、掩码和文本）生成高质量的对象掩码。此外，SAM可以为整个图像中的所有对象生成掩码。它构建了迄今为止最大的分割数据集[dataset](https://segment-anything.com/dataset/index.html) ，涵盖了超过1亿个掩码和1100万个经过授权且保护隐私的图像。SAM在各种任务的零样本性能表现出色，甚至经常与或甚至优于以前的全监督结果。

我们提供了PaddlePaddle格式的预训练模型参数，包括：[vit_b](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams), [vit_l](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams) 和 [vit_h](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams).

## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> 性能展示

<div align="center">
<img src="https://github.com/Sunting78/images/blob/master/sam_new.gif"  width="1000" />
</div>


## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> 由你自己来一行代码体验

### 1. 准备
* 安装PaddlePaddle和相关环境，可以参考 [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).
* 安装 PaddleSeg 基于 [reference](../../docs/install.md).
* 克隆 PaddleSeg 仓库:
    ```bash
    git clone https://github.com/PaddlePaddle/PaddleSeg.git
    cd PaddleSeg
    pip install -r requirements.txt
    ```
* 下载样例图像数据 ```contrib/SegmentAnything/examples```, 文件结构如下:
    ```bash
    wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
    ```

    ```
    PaddleSeg/contrib
    ├── SegmentAnything
    │   ├── examples
    │   │   └──  cityscapes_demo.png
    │   ├── segment_anything
    │   └── scripts
    ```

### 2. 在网页上对整个图像进行分割

在这一步骤中，我们使用以下脚本在本地机器上启动Gradio服务，您可以使用自己的图像尝试我们的项目。

1. 运行下面的脚本:
    ```bash
    python scripts/amg_paddle.py --model-type [vit_l/vit_b/vit_h] # default is vit_h
    ```
    注意:
    * 有三个模型选项可供选择，vit_b、vit_l和vit_h，分别代表vit_base、vit_large和vit_huge。大型模型更精确，但速度较慢。您可以根据设备的性能选择适合的模型尺寸。
    * 测试结果显示，vit_h需要16G的视频内存，并需要大约10秒钟在V100上推断一张图像。

2. 在本地打开网页: ```http://0.0.0.0:8017```

3. 清空并上传测试图像以尝试它！示例看起来如下图：:

    <div align="center">
    <img src="https://user-images.githubusercontent.com/34859558/230873989-9597527e-bef6-47ce-988b-977198794d75.jpg"  width = "1000" />  
    </div>

### 3. 使用提示分割对象

您可以运行以下命令，从不同类型的提示（包括点、框和掩码）生成掩码，如下所示：


1. 框 提示

```bash
python scripts/promt_predict.py --input_path xxx.png --box_prompt 1050 370 1500 700 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

2. 点 提示
```bash
python scripts/promt_predict.py --input_path xxx.png --point_prompt 1200 450 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

3. 掩码 提示
```bash
python scripts/promt_predict.py --input_path xxx.png --mask_prompt xxx.png --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

注意:
* mask_prompt是二进制图像的路径。
