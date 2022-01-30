cmake_minimum_required(VERSION 3.0)
project(onnxruntime_cpp_inference_demo CXX C)

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
)

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}_app
  ${PROJECT_SOURCE_DIR}/src/test_seg.cpp
  ${PROJECT_SOURCE_DIR}/src/ort_session_handler.cpp
)

target_link_libraries(${PROJECT_NAME}_app
  ${OpenCV_LIBRARIES}
  ${onnxruntime_LIBS}
)

target_include_directories(${PROJECT_NAME}_app
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${onnxruntime_INCLUDE_DIRS}
)
