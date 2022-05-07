#!/bin/bash
set +x
set -e

# 1. prepare
WITH_GPU=ON
USE_TENSORRT=ON
WITH_MKL=ON
DEMO_NAME=test_det

work_path=$(dirname $(readlink -f $0))
paddle_root="${work_path}/paddle_inference"
tensorrt_root='/work/download/TensorRT-7.1.3.4/'

model_dir='infer_models_det'
device=GPU
use_trt=True
trt_precision=fp32
use_trt_dynamic_shape=False
use_trt_auto_tune=True
warmup_iters=10
run_iters=20
save_path="res_det.txt"

#if [ ! -f "cityscapes_demo.png" ]; then
#  wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
#fi


echo "\n---Config Info---" >> ${save_path}
echo "device: ${device}" >> ${save_path}
echo "use_trt: ${use_trt}" >> ${save_path}
echo "trt_precision: ${trt_precision}" >> ${save_path}
echo "use_trt_dynamic_shape: ${use_trt_dynamic_shape}" >> ${save_path}
echo "use_trt_auto_tune: ${use_trt_auto_tune}" >> ${save_path}
echo "warmup_iters: ${warmup_iters}" >> ${save_path}
echo "run_iters: ${run_iters}" >> ${save_path}

echo "| model | preprocess time (ms) | run time (ms) |"  >> ${save_path}

# 2. compile
mkdir -p build_det
cd build_det
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
  echo "\n-----------------Test ${model}-----------------"
  ./build_det/test_det \
      --model_dir=${model} \
      --img_path=./000000014439.jpg \
      --device=${device} \
      --use_trt=${use_trt} \
      --trt_precision=${trt_precision} \
      --use_trt_dynamic_shape=${use_trt_dynamic_shape} \
      --use_trt_auto_tune=${use_trt_auto_tune} \
      --warmup_iters=${warmup_iters} \
      --run_iters=${run_iters} \
      --save_path=${save_path}
done
