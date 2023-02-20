[English](README.md) | 简体中文
# PP-LiteSeg 量化模型 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 PP-LiteSeg 量化模型在晶晨 A311D 上的部署推理加速。

## 1. 部署环境准备
### 1.1 FastDeploy 交叉编译环境准备
软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 晶晨 A311d 编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/a311d.md)  

## 2. 部署模型准备
1. 用户可以直接使用由[FastDeploy 提供的量化模型](../README_CN.md)进行部署。
2. 若FastDeploy没有提供满足要求的量化模型，用户可以参考[PaddleSeg动态图模型导出为A311D支持的INT8模型](../README_CN.md)自行导出或训练量化模型
3. 若上述导出或训练的模型出现精度下降或者报错，则需要使用异构计算，使得模型算子部分跑在A311D的ARM CPU上进行调试以及精度验证，其中异构计算所需的文件是subgraph.txt。具体关于异构计算可参考：[异构计算](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/heterogeneous_computing_on_timvx_npu.md)。

## 3. 在 A311D 上部署量化后的 PP-LiteSeg 分割模型
请按照以下步骤完成在 A311D 上部署 PP-LiteSeg 量化模型：

1. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ path/to/PaddleSeg/deploy/fastdeploy/semantic_segmentation/amlogic/a311d/cpp
```

2. 在当前路径下载部署所需的模型和示例图片：
```bash
cd path/to/PaddleSeg/deploy/fastdeploy/semantic_segmentation/amlogic/a311d/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz
tar -xvf ppliteseg.tar.gz
cp -r ppliteseg models
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
cp -r cityscapes_demo.png images
```

3. 编译部署示例，可使入如下命令：
```bash
cd path/to/PaddleSeg/deploy/fastdeploy/semantic_segmentation/amlogic/a311d/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

4. 基于 adb 工具部署 PP-LiteSeg 分割模型到晶晨 A311D，可使用如下命令：
```bash
# 进入 install 目录
cd path/to/paddleseg/amlogic/a311d/cpp/build/install/
cp ../../run_with_adb.sh .
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo ppliteseg cityscapes_demo.png $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/205544166-9b2719ff-ed82-4908-b90a-095de47392e1.png">

## 4. 更多指南
- [PaddleSeg C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1segmentation.html)
- [FastDeploy部署PaddleSeg模型概览](../../)
