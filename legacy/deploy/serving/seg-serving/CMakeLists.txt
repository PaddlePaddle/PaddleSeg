# TODO：下载data数据
#if (NOT EXISTS
#                ${CMAKE_CURRENT_LIST_DIR}/data/model/paddle/fluid/text_classification_lstm)
#        execute_process(COMMAND wget
#                --no-check-certificate https://paddle-serving.bj.bcebos.com/data/text_classification/text_classification_lstm.tar.gz
#                --output-document
#                ${CMAKE_CURRENT_LIST_DIR}/data/model/paddle/fluid/text_classification_lstm.tar.gz)

#           execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
#                "${CMAKE_CURRENT_LIST_DIR}/data/model/paddle/fluid/text_classification_lstm.tar.gz"
#                WORKING_DIRECTORY
#                ${CMAKE_CURRENT_LIST_DIR}/data/model/paddle/fluid
#                )
# endif()
include_directories(SYSTEM  ${CMAKE_CURRENT_LIST_DIR}/../kvdb/include)
find_library(MKLML_LIBS NAMES libmklml_intel.so libiomp5.so)
include(op/CMakeLists.txt)
include(proto/CMakeLists.txt)
add_executable(seg-serving ${serving_srcs})
add_dependencies(seg-serving pdcodegen fluid_cpu_engine pdserving paddle_fluid
        opencv_imgcodecs)
if (WITH_GPU)
    add_dependencies(seg-serving fluid_gpu_engine)
endif()
target_include_directories(seg-serving PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/../predictor
        )

if(WITH_GPU)
    target_link_libraries(seg-serving -Wl,--whole-archive fluid_gpu_engine
            -Wl,--no-whole-archive)
endif()

target_link_libraries(seg-serving -Wl,--whole-archive fluid_cpu_engine
        -Wl,--no-whole-archive)

target_link_libraries(seg-serving paddle_fluid ${paddle_depend_libs})

target_link_libraries(seg-serving opencv_imgcodecs
        ${opencv_depend_libs})

target_link_libraries(seg-serving pdserving)
target_link_libraries(seg-serving cube-api)
target_link_libraries(seg-serving kvdb rocksdb)

if(WITH_GPU)
    target_link_libraries(seg-serving ${CUDA_LIBRARIES})
endif()
target_link_libraries(seg-serving ${MKLML_LIB} ${MKLML_IOMP_LIB} -lpthread
        -lcrypto -lm -lrt -lssl -ldl -lz -lbz2)

install(TARGETS seg-serving
        RUNTIME DESTINATION
        ${PADDLE_SERVING_INSTALL_DIR}/demo/seg-serving/bin)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/conf DESTINATION
        ${PADDLE_SERVING_INSTALL_DIR}/demo/seg-serving/bin)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data DESTINATION
        ${PADDLE_SERVING_INSTALL_DIR}/demo/seg-serving/bin)

FILE(GLOB inc ${CMAKE_CURRENT_BINARY_DIR}/*.pb.h)
install(FILES ${inc}
        DESTINATION ${PADDLE_SERVING_INSTALL_DIR}/include/seg-serving)

if (${WITH_MKL})
        install(FILES ${THIRD_PARTY_PATH}/install/mklml/lib/libmklml_intel.so
                ${THIRD_PARTY_PATH}/install/mklml/lib/libmklml_gnu.so
                ${THIRD_PARTY_PATH}/install/mklml/lib/libiomp5.so DESTINATION
                ${PADDLE_SERVING_INSTALL_DIR}/demo/seg-serving/bin)
endif()
