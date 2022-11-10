wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xf 3.4.7.tar.gz

mkdir -p opencv-3.4.7/build
cd opencv-3.4.7/build

install_path=/usr/local/opencv3
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path} -DCMAKE_BUILD_TYPE=Release
make -j
make install

cd ../..
