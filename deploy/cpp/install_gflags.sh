git clone https://github.com/gflags/gflags.git
mkdir -p gflags/build
cd gflags/build
cmake ..
make -j
make install
