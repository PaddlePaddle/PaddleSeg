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
warmup_iters=20
run_iters=30
save_path="res.txt"

if [ ! -f "cityscapes_demo.png" ]; then
  wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
fi

if [ -f "${save_path}" ]; then
  rm -rf ${save_path}
  touch ${save_path}
fi

echo "---Config Info---" >> ${save_path}
echo "target_width: ${target_width}" >> ${save_path}
echo "target_height: ${target_height}" >> ${save_path}
echo "device: ${device}" >> ${save_path}
echo "use_trt: ${use_trt}" >> ${save_path}
echo "trt_precision: ${trt_precision}" >> ${save_path}
echo "use_trt_dynamic_shape: ${use_trt_dynamic_shape}" >> ${save_path}
echo "use_trt_auto_tune: ${use_trt_auto_tune}" >> ${save_path}
echo "warmup_iters: ${warmup_iters}" >> ${save_path}
echo "run_iters: ${run_iters}" >> ${save_path}
echo "\n" >> ${save_path}

echo "| model | preprocess time (ms) | run time (ms) |"  >> ${save_path}

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
  echo "\n-----------------Test ${model}-----------------\n"
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
      --run_iters=${run_iters} \
      --save_path=${save_path}
done
