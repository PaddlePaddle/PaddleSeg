# 依赖安装

## OpenCV
OpenCV官方Release地址：https://opencv.org/releases/

### Windows

1. 下载Windows安装包：OpenCV-3.4.6
2. 双击安装到指定位置，如D:\opencv
3. 配置环境变量  
> 1.我的电脑->属性->高级系统设置->环境变量  
> 2.在系统变量中找到Path（如没有，自行创建），并双击编辑  
> 3.新建，将opencv路径填入并保存，如D:\opencv\build\x64\vc14\bin  

### Linux
1. 下载OpenCV-3.4.6 Sources，并解压，如/home/user/opencv-3.4.6
2. cd opencv-3.4.6 & mkdir build & mkdir release
3. 修改modules/videoio/src/cap_v4l.cpp 在代码第253行下，插入如下代码
```
#ifndef V4L2_CID_ROTATE
#define V4L2_CID_ROTATE (V4L2_CID_BASE+34)
#endif
#ifndef V4L2_CID_IRIS_ABSOLUTE
#define V4L2_CID_IRIS_ABSOLUTE (V4L2_CID_CAMERA_CLASS_BASE+17)
#endif
```
3. cd build
4. cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/ssd2/Jason/tmp/opencv-3.4.6/release/ --OPENCV_FORCE_3RDPARTY_BUILD=OFF
5. make -j10
6. make install
编译后产出的头文件和lib即安装在/home/user/opencv-3.4.6/release目录下
