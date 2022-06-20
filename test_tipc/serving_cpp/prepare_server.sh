# 1. prepare serving source code
export https_proxy=${HTTP_PROXY} && export http_proxy=${HTTPS_PROXY}
git clone https://github.com/PaddlePaddle/Serving
cd Serving
git checkout -- .
git checkout -b v0.8.3 origin/v0.8.3
git submodule update --init --recursive
python3.7 -m pip install -r python/requirements.txt

# copy files to serving
cp -rf ../test_tipc/serving_cpp/general_seg_op.* core/general-server/op

# 2. prepare opencv
unset http_proxy && unset https_proxy
wget https://paddle-qa.bj.bcebos.com/PaddleServing/opencv3.tar.gz && tar -xf opencv3.tar.gz \
&& rm -rf opencv3.tar.gz && rm -rf /usr/local/opencv3 && mv opencv3 /usr/local
OPENCV_DIR=/usr/local/opencv3

# 3. prepare go
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
export https_proxy=${HTTP_PROXY} && export http_proxy=${HTTPS_PROXY}

go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct
go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
go install github.com/golang/protobuf/protoc-gen-go@v1.4.3
go install google.golang.org/grpc@v1.33.0
go env -w GO111MODULE=auto

# 4. set following variables according the docs
PYTHON_INCLUDE_DIR=/usr/local/include/python3.7m
PYTHON_LIBRARIES=/usr/local/lib/
PYTHON_EXECUTABLE=/usr/local/bin/python3.7

CUDA_PATH='/usr/local/cuda'
CUDNN_LIBRARY='/usr/local/cuda/lib64/'
CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-6.0.1.8/"

# 5. build paddle-serving-server
mkdir build_server_gpu_opencv_seg
cd build_server_gpu_opencv_seg
cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
    -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
    -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
    -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
    -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
    -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_OPENCV=ON \
    -DSERVER=ON \
    -DWITH_GPU=ON ..
make -j

# 6. install paddle-serving-server
python3.7 -m pip install python/dist/paddle*

unset http_proxy && unset https_proxy