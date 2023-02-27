[English](README.md) | 简体中文
# PaddleSeg Ascend NPU Python部署示例

本目录下提供`infer.py`快速完成PP-LiteSeg在华为昇腾上部署的示例。

## 1. 部署环境准备  
在部署前，需自行编译基于华为昇腾NPU的FastDeploy python wheel包并安装，参考文档[华为昇腾NPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 部署模型准备  
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleSeg部署模型](../README.md)，如果你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)。

## 3. 运行部署示例 
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
# 注意：如果当前分支找不到测试代码 请切换到develop分支
git checkout develop
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/ascend/python

# 下载PP-LiteSeg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 华为昇腾推理
python infer.py --model PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image cityscapes_demo.png
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

## 4. 更多指南
- [PaddleSeg python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/semantic_segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](..)
- [PaddleSeg C++部署](../cpp)

## 5. 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
