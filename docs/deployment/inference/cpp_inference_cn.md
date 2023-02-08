简体中文 | [English](cpp_inference.md)

# C++端预测部署总览

### 1. 各环境编译部署教程

* [Linux编译部署](cpp_inference_linux_cn.md)
* [Windows编译部署](cpp_inference_windows_cn.md)

### 2. 说明
`PaddleSeg/deploy/cpp`为用户提供了一个跨平台的C++部署方案，用户通过PaddleSeg训练的模型导出后，即可基于本项目快速运行，也可以快速集成代码结合到自己的项目实际应用中去。

主要设计的目标包括以下两点：

* 跨平台，支持在 Windows 和 Linux 完成编译、二次开发集成和部署运行
* 可扩展性，支持用户针对新模型开发自己特殊的数据预处理等逻辑

主要目录和文件说明如下：
```
deploy/cpp
|
├── cmake # 依赖的外部项目cmake（目前仅有yaml-cpp）
│
├── src ── test_seg.cc # 示例代码文件
│
├── CMakeList.txt # cmake编译入口文件
│
└── *.sh # Linux下安装相关包或运行示例脚本
```