English | [简体中文](whole_process_cn.md)
# Learn PaddleSeg in 20 mins

Paddleseg is composed of code modules and controls the setting use configs, in this way, it realize the full-process from data label to model deployment. In this tutorial, we will use `PP-LiteSeg model` and `Medical Video Disc Segmentation Dataset` as example to help you learn Paddleseg easily and quickily.

This tutorial contains the following steps:

1. **Prepare the environment**: PaddleSeg's software environment.
2. **Data preparation**: How to prepare and organize custom datasets.
3. **Model training**: Training configuration and start training.
4. **Visualize the training process**: Use VDL to show the training process.
5. **Model evaluation**: Evaluate the model.
6. **Model prediction and visualization**: Use the trained model to make predictions and visualize the results at the same time.
7. **Model export**: How to export a model that can be deployed.
8. **Model deployment**: Quickly use Python to achieve efficient deployment.

## **1. Environmental Installation**

### **1.1 Environment Installation**

Please refer to the [Installation Guide](./install.md) to prepare the environment.

## **2. Dataset Preparation**

### **2.1 Dataset Download**

Our demo uses the `optic disc segmentation dataset` for training. It is one of dozens datasets that we support. You can refer to [Datasets prepare](./data/pre_data.md) to use other datasets we support or [Prepare your own data](./data/marker/marker.md) to prepare your own data.

Optic disc segmentation is a set of fundus medical segmentation datasets, including 267 training images, 76 verification images, and 38 test images. You can download and unzip them using the following command.

```
mkdir data
cd data
wget https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
unzip optic_disc_seg.zip
cd ..
```

The original image and segmentation result are shown below. Our task will be to segment the optic disc area in the eyeball picture.

![](./images/fig1.png)


## **3. Model Training**

### **3.1 PP-LitSeg Model**

We choose the PP-LiteSeg model for training.

PP-LiteSeg model is a real-time semantic segmentation model, which is proposed by PaddleSeg team.
Compared to other models, PP-LiteSeg achieves superior trade-off between accuracy and speed on Cityscapes and CamVid dataset. Specifically, we present a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder. To strengthen feature representations, we propose a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost. The architecture of PP-LiteSeg is shown in next figure. For more information of PP-LiteSeg , please refer to [doc](../configs/pp_liteseg).


<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148786-c8b91fd1-d006-4bad-8599-556daf959a75.png" width = "600" height = "300" alt="arch"  />
</div>

### **3.2 Detailed Interpretation of Configuration Files**

After understanding the principle of PP-LiteSeg, we can prepare for training. In the above, we talked about PaddleSeg providing **configurable driver** for model training. So before training, let’s take a look at the configuration file. Here we take `pp_liteseg_optic_disc_512x512_1k.yml` as an example. The yaml format configuration file includes model type, backbone network, training and testing, pre-training dataset and supporting tools (such as Data augmentation) and other information.

PaddleSeg lists every option that can be optimized in the configuration file. Users can customize the model by modifying this configuration file (**All configuration files are under the PaddleSeg/configs folder**), such as custom models The backbone network used, the loss function used by the model, and the configuration of the network structure. In addition to customizing the model, data processing strategies can be configured in the configuration file, such as data augmentation strategies such as resizing, normalization, and flipping.

**Key Parameter:**

-1: In the learning rate given in the PaddleSeg configuration file, except for the single-card learning rate in "pp_liteseg_optic_disc_512x512_1k.yml", the rest of the configuration files are all 4-card learning rates. If the user is training with a single card, then learn The rate setting should become 1/4 of the original.
-2: The configuration file in PaddleSeg gives a variety of loss functions: CrossEntropy Loss, BootstrappedCrossEntropy Loss, Dice Loss, BCE Loss, OhemCrossEntropyLoss, RelaxBoundaryLoss, OhemEdgeAttentionLoss, Lovasz Hinge Loss, Lovasz Soft Loss, users can perform according to their own needs Change.
-3: The details of config file are as following.

```
batch_size: 4  # Set the number of pictures sent to the network at one iteration. Generally speaking, the larger the video memory of the machine you are using, the higher the batch_size value.
iters: 1000  # Number of iterations

train_dataset:  # Training dataset
  type: Dataset # The name of the training dataset class
  dataset_root: data/optic_disc_seg # The directory where the training dataset is stored
  train_path: data/optic_disc_seg/train_list.txt  # The list for training
  num_classes: 2  # Number of pixel categories
  mode: train # Indicate the dataset for training
  transforms: # Data transformation and data augmentation
    - type: ResizeStepScaling # Random resize the image and label to 0.5x~2.0x
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop # Random crop the resized image and label
      crop_size: [512, 512]
    - type: RandomHorizontalFlip  # Flip the image horizontally with a certain probability
    - type: RandomDistort # Change the brightness, contrast and saturation
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize # Normalize the image

val_dataset:  # Validating dataset
  type: Dataset # The name of the training dataset class
  dataset_root: data/optic_disc_seg # The directory where the validating dataset is stored
  val_path: data/optic_disc_seg/val_list.txt  # The list for validation
  num_classes: 2  # Number of pixel categories
  mode: val # Indicate the dataset for validating
  transforms: # Data transformation and data augmentation
    - type: Normalize # Normalize the image

optimizer: # Set the type of optimizer
  type: sgd #Using SGD (Stochastic Gradient Descent) method as the optimizer
  momentum: 0.9
  weight_decay: 4.0e-5 # Weight attenuation, the purpose of use is to prevent overfitting

lr_scheduler: # Related settings for learning rate
  type: PolynomialDecay # A type of learning rate,a total of 12 strategies are supported
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

loss: # Set the type of loss function
  types:
    - type: CrossEntropyLoss # The type of loss function
  coef: [1, 1, 1]
  # PP-LiteSeg has 2 auxiliary losses and a main losses, coef means weight: total_loss = coef_1 * loss_1 + .... + coef_n * loss_n

model:  # Model description
  type: PPLiteSeg  # Set model name
  backbone:  # Set the backbone，include name and pretrained weights
    type: STDC1
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz

```
**FAQ**

Q: Some readers may have questions, what kind of configuration items are designed in the configuration file, and what kind of configuration items are in the command line parameters of the script?

A: The information related to the model scheme is in the configuration file, and it also includes data augmentation strategies for the original sample. In addition to the three common parameters of iters, batch_size, and learning_rate, the command line parameters only involve the configuration of the training process. In other words, the configuration file ultimately determines what model to use.

### **3.3 Modify Configuration Files**

When the user prepares the dataset, he can specify the location in the configuration file to modify the data path for further training

Here, we take the "pp_liteseg_optic_disc_512x512_1k.yml" file mentioned in the above article as an example, and select the data configuration part for your explanation.

Mainly focus on these parameters:

- The type parameter is Dataset, which represents the recommended data format;
- The dataset_root path contains the path where the label and image are located; in the example: dataset_root: dataset/optic_disc_seg
- train_path is the path of txt; in the example: train_path: dataset/optic_disc_seg/train_list.txt
- num_classes is the category (the background is also counted as a category);
- Transform is a strategy for data preprocessing, users can change according to their actual needs

```
train_dataset:
  type: Dataset
  dataset_root: data/optic_disc_seg
  train_path: data/optic_disc_seg/train_list.txt
  num_classes: 2
  mode: train
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize

val_dataset:
  type: Dataset
  dataset_root: dataset/optic_disc_seg
  val_path: dataset/optic_disc_seg/val_list.txt
  num_classes: 2
  mode: val
  transforms:
    - type: Normalize
```

### **3.4 Start Training**

After we modify the corresponding configuration parameters, we can start training.

```
export CUDA_VISIBLE_DEVICES=0 # Set 1 usable card

**Please execute the following command under windows**
**set CUDA_VISIBLE_DEVICES=0**
python train.py \
        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
        --do_eval \
        --use_vdl \
        --save_interval 500 \
        --save_dir output
```

The weights of trained model is saved in `output`.

```
output
  ├── iter_500 # Means to save the model once at 500 steps
    ├── model.pdparams  # Model parameters
    └── model.pdopt  # Optimizer parameters during training
  ├── iter_1000
    ├── model.pdparams
    └── model.pdopt
  └── best_model # #During training, after training, add --do_eval, every time the model is saved, it will be evaled once, and the model with the highest miou will be saved as best_model
    └── model.pdparams  
```

### **3.5 Training Parameters**

| Parameter           | Effection                               | Is Required                   | Default           |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| iters               | Number of training iterations          | No         | The value specified in the configuration file.| |
| batch_size          | Batch size on a single card               | No         | The value specified in the configuration file.| |
| learning_rate       | Initial learning rate                | No        | The value specified in the configuration file.| |
| config              | Configuration files                                                     | Yes         | -                |
| save_dir            | The root path for saving model and visualdl log files           | No         | output           |
| num_workers         | The number of processes used to read data asynchronously, when it is greater than or equal to 1, the child process is started to read dat  | No  | 0 |
| use_vdl             | Whether to enable visualdl to record training data          | No         | No               |
| save_interval       | Number of steps between model saving             | No         | 1000             |
| do_eval             | Whether to do evaluation when saving the model, the best model will be saved according to mIoU | No   | No  |
| log_iters           | Interval steps for printing log          | No         | 10               |
| resume_model        | Restore the training model path, such as: `output/iter_1000`     | No        | None             |
| keep_checkpoint_max | Number of latest models saved                                            | No        | 5                |

### **3.6 In-depth Exploration of Configuration Files**

- We just took out a PP-LiteSeg configuration file for everyone to experience how to configure the dataset. In this example, all the parameters are placed in a yml file, but the actual PaddleSeg configuration file is for better reuse For compatibility and compatibility, a more coupled design is adopted, that is, a model requires more than two configuration files to achieve. Below we will use DeeplabV3p as an example to illustrate the coupling settings of the configuration files.
- For example, if we want to change the configuration of the deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml file, we will find that the file also depends on the (base) cityscapes.yml file. At this point, we need to open the cityscapes.yml file synchronously to set the corresponding parameters.

![](./images/fig3.png)

In PaddleSeg2.0 mode, users can find that PaddleSeg adopts a more coupled configuration design, placing common configurations such as data, optimizer, and loss function under a single configuration file. When we try to change to a new network The structure is time, you only need to pay attention to model switching, which avoids the tedious rhythm of switching models to re-adjust these common parameters and avoid user errors.

**FAQ**

Q: There are some common parameters in multiple configuration files, so which one shall I prevail?

A: As shown by the serial number in the figure, the parameters of the No. 1 yml file can cover the parameters of the No. 2 yml file, that is, the configuration file No. 1 is better than the No. 2. In addition, if the parameters appearing in the yaml file are specified in the command line, the configuration of the command line is better than the yaml file. (For example: adjust `batch_size` in the command line according to your machine configuration, no need to modify the preset yaml file in configs)

### **3.7 Muti-card Training**

**Note**: If you want to use multi-card training, you need to specify the environment variable `CUDA_VISIBLE_DEVICES` as `multi-card` (if not specified, all GPUs will be used by default), and use `paddle.distributed.launch` to start the training script (Can not use multi-card training under Windows, because it doesn't support nccl):

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set 4 usable cards
python -m paddle.distributed.launch train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### **3.8 Resume training**

```
python train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --resume_model output/iter_500 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

## **4. Training Process Visualization**

- In order to make our network training process more intuitive and analyze the network to get a better network faster, PaddlePaddle provides a visual analysis tool: VisualDL

When the `use_vdl` switch is turned on, PaddleSeg will write the data during the training process into the VisualDL file, and you can view the log during the training process in real time. The recorded data includes:

1. Loss change trend
2. Changes in learning rate
3. Training time
4. Data reading time
5. Mean IoU change trend (takes effect when the `do_eval` switch is turned on)
6. Change trend of mean pixel Accuracy (takes effect when the `do_eval` switch is turned on)

Use the following command to start VisualDL to view the log

```
**The following command will start a service on 127.0.0.1, which supports viewing through the front-end web page, and the actual ip address can be specified through the --host parameter**

visualdl --logdir output/
```

Enter the suggested URL in the browser, the effect is as follows:

![](./images/fig4.png)

## **5. Model Evaluation**

After the training is completed, the user can use the evaluation script val.py to evaluate the effect of the model. Assuming that the number of iterations (iters) in the training process is 1000, the interval for saving the model is 500, that is, the training model is saved twice for every 1000 iterations of the dataset. Therefore, there will be a total of 2 regularly saved models, plus the best model best_model saved, there are a total of 3 models. You can specify the model file you want to evaluate through model_path.

```
python val.py \
        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
        --model_path output/iter_1000/model.pdparams
```

If you want to perform multi-scale flip evaluation, you can turn it on by passing in `--aug_eval`, and then passing in scale information via `--scales`, `--flip_horizontal` turns on horizontal flip, and `flip_vertical` turns on vertical flip. Examples of usage are as follows:

```
python val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --aug_eval \
       --scales 0.75 1.0 1.25 \
       --flip_horizontal
```

If you want to perform sliding window evaluation, you can open it by passing in `--is_slide`, pass in the window size by `--crop_size`, and pass in the step size by `--stride`. Examples of usage are as follows:

```
python val.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams \
       --is_slide \
       --crop_size 256 256 \
       --stride 128 128
```

In the field of image segmentation, evaluating model quality is mainly judged by three indicators, `accuracy` (acc), `mean intersection over union` (mIoU), and `Kappa coefficient`.

- **Accuracy**: refers to the proportion of pixels with correct category prediction to the total pixels. The higher the accuracy, the better the quality of the model.
- **Average intersection ratio**: perform inference calculations for each category dataset separately, divide the calculated intersection of the predicted area and the actual area by the union of the predicted area and the actual area, and then average the results of all categories. In this example, under normal circumstances, the mIoU index value of the model on the verification set will reach 0.80 or more. An example of the displayed information is shown below. The **mIoU=0.8526** in the third row is mIoU.
- **Kappa coefficient**: an index used for consistency testing, which can be used to measure the effect of classification. The calculation of the kappa coefficient is based on the confusion matrix, with a value between -1 and 1, usually greater than 0. The formula is as follows, P0P_0*P*0 is the accuracy of the classifier, and PeP_e*P**e* is the accuracy of the random classifier. The higher the Kappa coefficient, the better the model quality.

<a href="https://www.codecogs.com/eqnedit.php?latex=Kappa=&space;\frac{P_0-P_e}{1-P_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Kappa=&space;\frac{P_0-P_e}{1-P_e}" title="Kappa= \frac{P_0-P_e}{1-P_e}" /></a>

With the running of the evaluation script, the final printed evaluation log is as follows.

```
...
2021-01-13 16:41:29 [INFO]    Start evaluating (total_samples=76, total_iters=76)...
76/76 [==============================] - 2s 30ms/step - batch_cost: 0.0268 - reader cost: 1.7656e-
2021-01-13 16:41:31 [INFO]    [EVAL] #Images=76 mIoU=0.8526 Acc=0.9942 Kappa=0.8283
2021-01-13 16:41:31 [INFO]    [EVAL] Class IoU:
[0.9941 0.7112]
2021-01-13 16:41:31 [INFO]    [EVAL] Class Acc:
[0.9959 0.8886]
```

## **6.Prediction and Visualization**

In addition to analyzing the IOU, ACC and Kappa indicators of the model, we can also check the cutting sample effect of some specific samples, and inspire further optimization ideas from Bad Case.

The predict.py script is specially used to visualize prediction cases. The command format is as follows

```
python predict.py \
        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
        --model_path output/iter_1000/model.pdparams \
        --image_path dataset/optic_disc_seg/JPEGImages/H0003.jpg \
        --save_dir output/result
```

Among them, `image_path` can also be a directory. At this time, all the pictures in the directory will be predicted and the visualization results will be saved.

Similarly, you can use `--aug_pred` to turn on multi-scale flip prediction, and `--is_slide` to turn on sliding window prediction.

We select 1 picture to view, the effect is as follows. We can intuitively see the difference between the cutting effect of the model and the original mark, thereby generating some optimization ideas, such as whether the cutting boundary can be processed in a regular manner.

![](./images/fig5.png)

## **7 Model Export**

In order to facilitate the user's industrial-level deployment, PaddleSeg provides a one-click function of moving to static, which is to convert the trained dynamic graph model file into a static graph form.

```
python export.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --model_path output/iter_1000/model.pdparams
```

- Parameters

| Parameter     | Effection                               | Is Required | Default           |
| :--------- | :--------------------------------- | :--------- | :--------------- |
| config     | Configuration file                           | Yes         | -                |
| save_dir   | The root path for saving model and visualdl log files | No         | output           |
| model_path | Path of pretrained model parameters             | No        | The value specified in the configuration file. |

```
- Result Files

output
  ├── deploy.yaml            # Deployment related configuration files
  ├── model.pdiparams        # Static graph model parameters
  ├── model.pdiparams.info   # Additional parameter information, generally don’t need attention
  └── model.pdmodel          # Static graph model files
```

## **8 Model Deploy**

-PaddleSeg currently supports the following deployment methods:

| Platform         | Library           | Tutorial  |
| :----------- | :----------- | :----- |
| Python | Paddle prediction library | [e.g.](../deploy/python/) |
| C++ | Paddle prediction library | [e.g.](../deploy/cpp/) |
| Mobile | PaddleLite   | [e.g.](../deploy/lite/) |
| Serving | HubServing   | Comming soon |
| Front-end | PaddleJS     | [e.g.](../deploy/web/) |

```
#Run the following command, an image of H0003.png will be generated under the output file
python deploy/python/infer.py \
--config output/deploy.yaml\
--image_path dataset/optic_disc_seg/JPEGImages/H0003.jpg\
--save_dir output
```

- Parameters:

|Parameter|Effection|Is required|Default|
|-|-|-|-|
|config|**Configuration file generated when exporting the model**, instead of the configuration file in the configs directory|Yes|-|
|image_path|The path or directory of the test image.|Yes|-|
|use_trt|Whether to enable TensorRT to accelerate prediction.|No|No|
|use_int8|Whether to run in int8 mode when starting TensorRT prediction.|No|No|
|batch_size|Batch sizein single card.|No|The value specified in the configuration file.|
|save_dir|The directory of prediction results.|No|output|
|with_argmax|Perform argmax operation on the prediction results.|No|No|

## **9 Custom Software Development**

- After trying to complete the training with the configuration file, there must be some friends who want to develop more in-depth development based on PaddleSeg. Here, we will briefly introduce the code structure of PaddleSeg.

```
PaddleSeg
     ├── configs # Configuration file folder
     ├── paddleseg # core code for training deployment
        ├── core # Start model training, evaluation and prediction interface
        ├── cvlibs # The Config class is defined in this folder. It saves all hyperparameters such as dataset, model configuration, backbone network, loss function, etc.
            ├── callbacks.py
            └── ...
        ├── datasets # PaddleSeg supported data formats, including ade, citycapes and other formats
            ├── ade.py
            ├── citycapes.py
            └── ...
        ├── models # This folder contains the various parts of the PaddleSeg network
            ├── backbone # The backbone network used by paddleseg
            ├── hrnet.py
            ├── resnet_vd.py
            └── ...
            ├── layers # Some components, such as the attention mechanism
            ├── activation.py
            ├── attention.py
            └── ...
            ├── losses # This folder contains the loss function used by PaddleSeg
            ├── dice_loss.py
            ├── lovasz_loss.py
            └── ...
            ├── ann.py # This file represents the algorithm model supported by PaddleSeg, here represents the ann algorithm.
            ├── deeplab.py #This file represents the algorithm model supported by PaddleSeg, here it represents the Deeplab algorithm.
            ├── unet.py #This file represents the algorithm model supported by PaddleSeg, here it represents the unet algorithm.
            └── ...
        ├── transforms # Data preprocessing operations, including various data augmentation strategies
            ├── functional.py
            └── transforms.py
        └── utils
            ├── config_check.py
            ├── visualize.py
            └── ...
     ├── train.py # The training entry file, which describes the analysis of parameters, the starting method of training, and the resources prepared for training.
     ├── predict.py # Prediction file
     └── ...


```

- You can also try to use PaddleSeg's API to develop themselves. After installing PaddleSeg using the pip install command, developers can easily implement the training, evaluation and inference of the image segmentation model with just a few lines of code. Interested friends can visit [PaddleSeg dynamic graph API usage tutorial](https://aistudio.baidu.com/aistudio/projectdetail/1339458?channelType=0&channel=0)

PaddleSeg and other development kits in various fields have provided top-level solutions for real industrial practice. Some domestic teams have used PaddleSeg's development kits to achieve good results in international competitions. It can be seen that the effects provided by the development kits are State Of The Art.
