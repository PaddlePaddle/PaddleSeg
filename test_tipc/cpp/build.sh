#!/bin/bash
set +x
set -e


work_path=$(dirname $(readlink -f $0))
LIB_DIR="${work_path}/paddle_inference"
echo $LIB_DIR
OPENCV_DIR="${work_path}/opencv-3.4.7/opencv3/"
echo $OPENCV_DIR
CUDA_LIB_DIR='/usr/local/cuda/lib64'
CUDNN_LIB_DIR='/usr/lib64'
TENSORRT_DIR=''

# compile
BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DWITH_STATIC_LIB=OFF \
    -DDEMO_NAME=run_seg \

make -j
