#!/bin/bash
set +x
set -e


WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
DEMO_NAME=run_seg

work_path=$(dirname $(readlink -f $0))
LIB_DIR="${work_path}/paddle_inference"
OPENCV_DIR='/usr/local/opencv3'
CUDA_LIB_DIR='/usr/local/cuda/lib64'
CUDNN_LIB_DIR='/usr/lib64'
TENSORRT_ROOT='/work/download/TensorRT-7.1.3.4/'

# compile
BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DDEMO_NAME=run_seg \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
