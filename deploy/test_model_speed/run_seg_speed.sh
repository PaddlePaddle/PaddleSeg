#!/bin/bash
set +x
set -e

# 1. prepare
WITH_GPU=ON
USE_TENSORRT=ON
WITH_MKL=ON
DEMO_NAME=test_seg

work_path=$(dirname $(readlink -f $0))
paddle_root="${work_path}/paddle_inference"
tensorrt_root='/work/download/TensorRT-7.1.3.4/'

model_dir='infer_models'
target_width=512
target_height=512
device=GPU
use_trt=True
trt_precision=fp32
use_trt_dynamic_shape=True
use_trt_auto_tune=True
warmup_iters=10
run_iters=20

if [ ! -f "cityscapes_demo.png" ]; then
  wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
fi

# 2. compile
mkdir -p build
cd build
rm -rf *

cmake .. \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_STATIC_LIB=OFF \
  -DPADDLE_LIB=${paddle_root} \
  -DTENSORRT_ROOT=${tensorrt_root}

make -j

# 3. run
cd ..
for model in ${model_dir}/*
do
  echo "-----------------Test ${model}-----------------"
  ./build/test_seg \
      --model_dir=${model} \
      --img_path=./cityscapes_demo.png \
      --target_width=${target_width} \
      --target_height=${target_height} \
      --device=${device} \
      --use_trt=${use_trt} \
      --trt_precision=${trt_precision} \
      --use_trt_dynamic_shape=${use_trt_dynamic_shape} \
      --use_trt_auto_tune=${use_trt_auto_tune} \
      --warmup_iters=${warmup_iters} \
      --run_iters=${run_iters}
  echo "\n"
done
