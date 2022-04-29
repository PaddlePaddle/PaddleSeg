English | [简体中文](prune_cn.md)

# Model Pruning Toturial

Many CNN models require huge computing and memory overhead, which seriously hinders the applications under limited resources. Model compression can reduce model parameters or FLOPs, and facilitate the deployment of restricted hardware. Powered by PaddleSlim, PaddleSeg provides model pruning for developers in image segmentation.

## Install PaddleSlim

Before model pruning, please install dependencies：

```shell
pip install paddleslim==2.0.0
```

## Model Pruning

Model pruning is one of model compression techniques, in which it reduces model size and computing complexity by reducing the number of kernels in the convolution layer. Based on PaddleSlim, PaddleSeg provides a sensitivity-based channel pruning method. The method can quickly analyze redundant paramenters in the model, and prune the model according to the pruning ratio specified by the user, which achieve a better trade-off between accuracy and speed.

*Note: So far only the following models support pruning, and more models are being supporting soon：*
*BiSeNetv2、FCN、Fast-SCNN、HardNet、UNet*

### 1. Model Training

We can train the model through the script provided by PaddleSeg. Please make sure that the installation of PaddleSeg is completed, and it is located in the PaddleSeg directory. Run the following script:：


```shell
export CUDA_VISIBLE_DEVICES=0 # Set an available gpu card.
# if Windows, set CUDA_VISIBLE_DEVICES=0
python train.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### 2. Model Pruning and Saving

Load the model trained in the previous step, specify the pruning rate, and start the script.

*Note: The sensitivity-based channel clipping method needs to continuously evaluate the impact of each convolution kernel on the final accuracy, so it will take a long time.*

|Parameters|Meaning|Required|Defaults|
|-|-|-|-|
|pruning_ratio|the convolution kernel pruning ratio|Yes||
|retraining_iters|retraining iters after pruning|Yes||
|config|configuration file|Yes||
|batch_size|batch size per gpu when retraining|No|specified in the configuration file|
|learning_rate|learning rate when retraining|No|specified in the configuration file|
|model_path|pretrained model parameters path|No||
|num_workers|multi-processing workers for reading data|No|0|
|save_dir|the save path of pruned model|No|output|

```shell
# Run the following command in root dir of PaddleSeg
export PYTHONPATH=`pwd`
# if windows, run the following command:
# set PYTHONPATH=%cd%

python slim/prune/prune.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --pruning_ratio 0.2 \
       --model_path output/best_model/model.pdparams \
       --retraining_iters 100 \
       --save_dir prune_model
```

## Deployment

The pruned model can be deployed directly. Please refer to the [tutorial](../../model_export.md) for model deployment.


## Pruning Speedup

Testing enviornment：
* GPU: V100
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* CUDA: 10.2
* cuDNN: 7.6
* TensorRT: 6.0.1.5

Testing method:
1. The running time is only model prediction time，and testing image is from Cityspcaes (1024x2048).
2. Predict 10 times as warm-up, and take the average time of 50 consecutive predictions.
3. Test with GPU + TensorRT.

|model|pruning ratio|execution time (ms)|speedup ratio|
|-|-|-|-|
|fastscnn|-|7.0|-|
||0.1|5.9|15.71%|
||0.2|5.7|18.57%|
||0.3|5.6|20.00%|
|fcn_hrnetw18|-|43.28|-|
||0.1|40.46|6.51%|
||0.2|40.41|6.63%|
||0.3|38.84|10.25%|
|unet|-|76.04|-|
||0.1|74.39|2.16%|
||0.2|72.10|5.18%|
||0.3|66.96|11.94%|
