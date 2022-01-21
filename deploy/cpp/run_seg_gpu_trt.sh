#!/bin/bash
set +x
set -e

# set TENSORRT_ROOT
TENSORRT_ROOT='/work/download/TensorRT-7.1.3.4/'

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
DEMO_NAME=test_seg

work_path=$(dirname $(readlink -f $0))
LIB_DIR="${work_path}/paddle_inference"

# compile
mkdir -p build
cd build
rm -rf *

cmake .. \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_STATIC_LIB=OFF \
  -DPADDLE_LIB=${LIB_DIR} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j

# run
cd ..

./build/test_seg \
    --model_dir=./stdc1seg_infer_model \
    --img_path=./cityscapes_demo.png \
    --devices=GPU \
    --use_trt=True \
    --trt_precision=fp32
