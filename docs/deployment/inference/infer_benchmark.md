English|[简体中文](infer_benchmark_cn.md)
# Inference Benchmark

Test Environment：
* GPU: V100 32G
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* CUDA: 10.1
* cuDNN: 7.6
* TensorRT: 6.0.1.5
* Paddle: 2.1.1

The method of test segmentation model on GPU:
1. Use all of the data in Cityscapes dataset to test(1024 * 2048).
2. Use single GPU and set batchsize to 1.
3. The time only includes model inference.
4. Use the [Python API](./python_inference.md) of Paddle Inference to test. You can choose whether to use TRT wirh use_trt parameter and use precision to set the inference datatype.

Inference with GPU Benchmark：

|       Model         |  With TRT   |   infer datatype  |  mIoU  |   time(s/img)   |
|        -                 |   :-:      |   :-:     |   :-:   |   :-:           |
| ANN_ResNet50_OS8         |   N        |    FP32    |  0.7909  |  0.274  |  
| ANN_ResNet50_OS8         |   Y        |    FP32    |  0.7909  |  0.281  |
| ANN_ResNet50_OS8         |   Y        |    FP16    |  0.7909  |  0.168  |
| ANN_ResNet50_OS8         |   Y        |    INT8    |  0.7906  |  0.195  |
| DANet_ResNet50_OS8         |   N        |    FP32    |  0.8027  |  0.371  |  
| DANet_ResNet50_OS8         |   Y        |    FP32    |  0.8027  |  0.330  |
| DANet_ResNet50_OS8         |   Y        |    FP16    |  0.8027  |  0.183  |
| DANet_ResNet50_OS8         |   Y        |    INT8    |  0.8039  |  0.266  |
| DeepLabV3P_ResNet50_OS8         |   N        |    FP32    |  0.8036  | 0.165  |  
| DeepLabV3P_ResNet50_OS8         |   Y        |    FP32    |  0.8036  | 0.206  |
| DeepLabV3P_ResNet50_OS8         |   Y        |    FP16    |  0.8036  | 0.196  |
| DeepLabV3P_ResNet50_OS8         |   Y        |    INT8    |  0.8044  | 0.083  |
| DNLNet_ResNet50_OS8         |   N        |    FP32    |  0.7995  |  0.381  |  
| DNLNet_ResNet50_OS8         |   Y        |    FP32    |  0.7995  |  0.360  |
| DNLNet_ResNet50_OS8         |   Y        |    FP16    |  0.7995  |  0.230  |
| DNLNet_ResNet50_OS8         |   Y        |    INT8    |  0.7989  |  0.236  |
| EMANet_ResNet50_OS8         |   N        |    FP32    |  0.7905  |  0.208  |  
| EMANet_ResNet50_OS8         |   Y        |    FP32    |  0.7905  |  0.186  |
| EMANet_ResNet50_OS8         |   Y        |    FP16    |  0.7904  |  0.062  |
| EMANet_ResNet50_OS8         |   Y        |    INT8    |  0.7939  |  0.106  |
| GCNet_ResNet50_OS8         |   N        |    FP32    |  0.7950  |  0.247  |  
| GCNet_ResNet50_OS8         |   Y        |    FP32    |  0.7950  |  0.228  |
| GCNet_ResNet50_OS8         |   Y        |    FP16    |  0.7950  |  0.100  |
| GCNet_ResNet50_OS8         |   Y        |    INT8    |  0.7959  |  0.144  |
| PSPNet_ResNet50_OS8         |   N        |    FP32    |  0.7883 | 0.327  |
| PSPNet_ResNet50_OS8         |   Y        |    FP32    |  0.7883 | 0.324  |
| PSPNet_ResNet50_OS8         |   Y        |    FP16    |  0.7883 | 0.218  |
| PSPNet_ResNet50_OS8         |   Y        |    INT8    |  0.7915 | 0.223  |
| UNet         |   N        |    FP32    |  0.6500  |  0.071  |  
| UNet         |   Y        |    FP32    |  0.6500  |  0.099  |
| UNet         |   Y        |    FP16    |  0.6500  |  0.099  |
| UNet         |   Y        |    INT8    |  0.6503  |  0.099  |
