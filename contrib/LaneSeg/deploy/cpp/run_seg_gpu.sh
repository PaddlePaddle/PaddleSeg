#!/bin/bash
set +x
set -e

WITH_MKL=ON
WITH_GPU=ON
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
# change model_dir and img_path according to your needs
./build/test_seg \
    --model_dir=../../output/export/ \
    --img_path=../../data/test_images/3.jpg \
    --use_cpu=false
