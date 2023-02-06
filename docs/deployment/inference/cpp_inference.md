[简体中文](cpp_inference_cn.md) | English

# C++ prediction deployment overview

### 1. Compilation and deployment tutorial for different environment

* [Compilation and deployment on Linux](cpp_inference_linux.md)
* [Compilation and deployment on Windows](cpp_inference_windows.md)

### 2. Illustration
`PaddleSeg/deploy/cpp` provides users with a cross-platform C++deployment scheme. After exporting the PaddleSeg training model, users can quickly run based on the project, or quickly integrate the code into their own project application.
The main design objectives include the following two points:

* Cross-platform, supporting compilation, secondary development integration and deployment on Windows and Linux
* Extensibility, supporting users to develop their own special data preprocessing and other logic for the new model

The main directory and documents are described as follows:
```
deploy/cpp
|
├── cmake # Dependent external project cmake (currently only yaml-cpp)
│
├── src ── test_seg.cc # Sample code file
│
├── CMakeList.txt # Cmake compilation entry file
│
└── *.sh # Install related packages or run sample scripts under Linux
```