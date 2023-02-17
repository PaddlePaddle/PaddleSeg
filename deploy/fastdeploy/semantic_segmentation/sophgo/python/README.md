[English](README.md) | 简体中文
# PaddleSeg Python部署示例

## 算能硬件编译FastDeploy wheel包环境准备

在部署前，需自行编译基于算能硬件的FastDeploy python wheel包并安装，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

本目录下提供`infer.py`快速完成 pp_liteseg 在SOPHGO TPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd path/to/paddleseg/sophgo/python

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# PaddleSeg模型转换为bmodel模型
将Paddle模型转换为SOPHGO bmodel模型，转换步骤参考[文档](../README_CN.md#将paddleseg推理模型转换为bmodel模型步骤)

# 推理
python3 infer.py --model_file ./bmodel/pp_liteseg_1684x_f32.bmodel --config_file ./bmodel/deploy.yaml --image cityscapes_demo.png

# 运行完成后返回结果如下所示
运行结果保存在sophgo_img.png中
```

## 快速链接
- [pp_liteseg C++部署](../cpp)
- [转换 pp_liteseg SOPHGO模型文档](../README_CN.md#导出bmodel模型)

## 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
