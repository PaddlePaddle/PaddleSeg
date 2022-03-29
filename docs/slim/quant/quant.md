# Model Quantization Tutorial


## 1. Introduction

Model quantization uses low bit values to replace high bit values and it is an amazing compression method.

For example, if float values is repleaced by int8 values, the size of the model can be reduced by 4 time and the inference speed can be accelerated.

Based on PaddleSlim, PaddleSeg supports quantization aware training method (QAT). The features of QAT are as follows:
* Use the train dataset to minimize the quantization error.
* Pros: The accuracy of the quantized model and the original model are similar.
* Cons: It takes a long time to train a quantized model.

## 2. Compare Accuracy and Performance

Requirements:
* GPU: V100 32G
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* CUDA: 10.1
* cuDNN: 7.6
* TensorRT: 6.0.1.5
* Paddle: 2.1.1

Details:
* Run the original model and quantized model on Nvidia GPU and enable TensorRT.
* Use one Nvidia GPU and the batch size is 1.
* Use the test dataset of Cityscapes with the size of 1024*2048.
* Only count the cost time of running predictor.


The next table shows the accuracy and performance of the original model and quantized model.

| Model | Dtype | mIoU |  Time(s/img） | Ratio |
| - | :-: | :-: | :-: | :-: |
| ANN_ResNet50_OS8 | FP32 | 0.7909  |  0.281  | - |
| ANN_ResNet50_OS8 | INT8 | 0.7906  |  0.195  | 30.6% |
| DANet_ResNet50_OS8 | FP32 | 0.8027  |  0.330  | - |
| DANet_ResNet50_OS8 | INT8 | 0.8039  |  0.266  | 19.4% |
| DeepLabV3P_ResNet50_OS8 | FP32 | 0.8036  | 0.206  |  - |  
| DeepLabV3P_ResNet50_OS8 | INT8 | 0.8044  | 0.083  | 59.7% |
| DNLNet_ResNet50_OS8 | FP32 | 0.7995  |  0.360  |  - |
| DNLNet_ResNet50_OS8 | INT8 | 0.7989  |  0.236  | 52.5% |
| EMANet_ResNet50_OS8 | FP32 |  0.7905  |  0.186  |  - |
| EMANet_ResNet50_OS8 | INT8 | 0.7939  |  0.106  | 43.0% |
| GCNet_ResNet50_OS8 | FP32 | 0.7950  |  0.228  |  - |
| GCNet_ResNet50_OS8 | INT8 | 0.7959  |  0.144  | 36.8% |
| PSPNet_ResNet50_OS8 | FP32 | 0.7883 | 0.324  |  - |
| PSPNet_ResNet50_OS8 | INT8 | 0.7915 | 0.223  | 32.1% |

## 3. Model Quantization Demo

We use a demo to explain how to generate and deploy a quantized model.

### 3.1 Preparation

Please refer to the [installation document](../../install.md) and prepare the requirements of PaddleSeg.
Note that, the quantization module requires the version of PaddlePaddle is at least 2.2.

Run the following instructions to install PaddleSlim.

```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git

# checkout to special commit
git reset --hard 15ef0c7dcee5a622787b7445f21ad9d1dea0a933

# install
python setup.py install
```

### 3.2 Generate Quantized Model

#### 3.2.1 Training for the Original Model

Before generating the quantized model, we have to prepare the original model with the data type of FP32.

In this demo, we choose the BiseNetV2 model and the optic disc segmentation dataset, and use `train.py` for training from scratch.
The usage of `train.py` can be found in this [document](../../train/train.md).

Specifically, run the following instructions in the root directory of PaddleSeg.

```shell
export CUDA_VISIBLE_DEVICES=0  # Set GPU for Linux
# set CUDA_VISIBLE_DEVICES=0   # Seg GPU for Windows

python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --save_dir output_fp32
```

After the training, the original model with the highest accuracy will be saved in `output_fp32/best_model`.

#### 3.2.2 Generate the Quantized Model

**1) Generate the Quantized Model**

Based on the original model, we use `slim/quant/qat_train.py` to generate the quantized model.

The usage of `qat_train.py` and `train.py` is basically the same, and the former uses `model_path` to set the weight path of the original model (as follows). Besides, the learning rate of the quantization training is usually smaller than the normal training.

| Input Params        | Usage                                                        | Optional   | Default Value          |
| ------------------- | ------------------------------------------------------------ | ---------- | ----------------  |
| config              | The config path of the original model                        | No         |     -             |
| model_path          | The path of weight of the original model                     | No         |     -             |
| iters               | Iterations                                                   | Yes        | The iters in config         |
| batch_size          | Batch size for single GPU                                    | Yes        | The batch_size in config    |
| learning_rate       | Learning rate                                                | Yes        | The learning_rate in config |  
| save_dir            | The directory for saving model and logs                      | Yes        | output           |
| num_workers         | The nums of threads to processs images                       | Yes        | 0                |
| use_vdl             | Whether to enable visualdl                                   | Yes        | False            |
| save_interval_iters | The interval interations for saving                          | Yes        | 1000             |
| do_eval             | Enable evaluation in training stage                          | Yes        | False            |
| log_iters           | The interval interations for outputing log                   | Yes        | 10               |
| resume_model        | The resume path, such as：`output/iter_1000`                  | Yes       | None             |


Run the following instructions in the root directory of PaddleSeg to start the quantization training.
After the quantization training, the quantized model with the highest accuracy will be saved in `output_quant/best_model`.

```shell
python slim/quant/qat_train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_fp32/best_model/model.pdparams \
       --learning_rate 0.001 \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --save_dir output_quant
```

**2）Test the Quantized Model (Optional)**

We use `slim/quant/qat_val.py` to load the weights of the quantized model and test the accuracy.

```
python slim/quant/qat_val.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams
```

**3）Export the Quantized Model**

Before deploying the quantized model, we have to convert the dygraph model to the inference model.

With the weights of the quantized model, we utilize `slim/quant/qat_export.py` to export the inference model.
The input params of the script are as follows.

|Input params| Usage | Optional | Default Value|
|-|-|-|-|
|config         | The config path of the original model                               | No  |  -                       |
|save_dir       | The save directory for saving the inference model                   | Yes | ./output                 |
|model_path     | The path of the quantized weights                                   | Yes | The model_path in config |
|with_softmax   | Whether to add softmax layer to the last of the inference model     | Yes | False                    |
|without_argmax | Whether not to add argmax layer to the last of the inference mode   | Yes | False                    |

Run the following instructions in the root directory of PaddleSeg. Then, the quantized inference model will be saved in `output_quant_infer`.

```
python slim/quant/qat_export.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path output_quant/best_model/model.pdparams \
       --save_dir output_quant_infer
```

### 3.3 Deploy the Quantized Model

We deploy the quantized inference model on Nvidia GPU and X86 CPU with Paddle Inference.
Besides, Paddle Lite support deploying the quantized model on ARM CPU.

Please refer to the documents for detail information:
* [Paddle Inference Python Deployment](../../deployment/inference/python_inference.md)
* [Paddle Inference C++ Deployment](../../deployment/inference/cpp_inference.md)
* [PaddleLite Deployment](../../deployment/lite/lite.md)

## 4. Reference

* [PaddleSlim Github](https://github.com/PaddlePaddle/PaddleSlim)
* [PaddleSlim Documents](https://paddleslim.readthedocs.io/zh_CN/latest/)
