#!/bin/bash
set +x
set -e

WITH_MKL=ON
WITH_GPU=OFF
USE_TENSORRT=OFF
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
  -DPADDLE_LIB=${LIB_DIR}

make -j

# run
cd ..

./build/test_seg \
    --model_dir=./pp_liteseg_infer_model \
    --img_path=./cityscapes_demo.png \
    --devices=CPU \
    --use_mkldnn=true
