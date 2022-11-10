git clone https://github.com/google/glog
mkdir -p glog/build
cd glog/build
cmake ..
make -j
make install
