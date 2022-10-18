wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
unzip yaml-cpp-0.7.0.zip
mkdir -p yaml-cpp-yaml-cpp-0.7.0/build
cd yaml-cpp-yaml-cpp-0.7.0/build
cmake -DYAML_BUILD_SHARED_LIBS=ON ..
make -j
make install
