[English](README.md) | 简体中文
# PaddleSeg Python部署示例

本目录下提供`infer.py`快速完成PP-LiteSeg在华为昇腾上部署的示例。

## 昆仑XPU编译FastDeploy wheel包环境准备

在部署前，需自行编译基于昆仑XPU的FastDeploy python wheel包并安装，参考文档[昆仑芯XPU部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

>>**注意** **PP-Matting**、**PP-HumanMatting**的模型，请从[Matting模型部署](../../../ppmatting)下载


```bash
#下载部署示例代码
cd path/to/paddleseg/ascend/cpp

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

## 快速链接
- [PaddleSeg python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/semantic_segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](..)
- [PaddleSeg C++部署](../cpp)

## 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
