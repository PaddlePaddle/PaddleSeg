# MedicalSeg
Welcome to MedicalSeg! MedicalSeg is an easy-to-use 3D medical image segmentation toolkit that supports GPU acceleration from data preprocess to deply. We aim to build our toolkit to support various datasets including lung, brain, and spine. (Currently contains [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) dataset and [MRISpineSeg](https://aistudio.baidu.com/aistudio/datasetdetail/81211).)

## 0. Model performance

###  1) Accuracy

We successfully validate our framework with [Vnet](https://arxiv.org/abs/1606.04797) on the [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans) dataset. With the lung mask as label, we reached dice coefficient of 97.04%. You can download the log to see the result or load the model and validate it by yourself :).

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|128x128x128|0.001|15000|97.04%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9db5c1e11ebc82f9a470f01a9114bd3c)|
|-|128x128x128|0.0003|15000|92.70%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0fb90ee5a6ea8821c0d61a6857ba4614)|


The segmentation result of our vnet model is presented as follows thanks to the powerful 3D visualization toolkit [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets). You can try to play around using our [visualize.ipynb](visualize.ipynb)

<p align="center">
<img src="https://github.com/shiyutang/files/blob/main/lung_prediction.gif?raw=true" width="90%" height="90%">
</p>

### 2) Speed
We add gpu acceleration in data preprocess using [CuPy](https://docs.cupy.dev/en/stable/index.html). Compared with preprocess data on cpu, acceleration enable us to use about 40% less time in data prepeocessing. The following shows the time we spend in process COVID-19 CT scans.

<center>

| Device | Time(s) |
|:-:|:-:|
|CPU|50.7|
|GPU|31.4( &#8595; 38%)|

</center>


## 1. Run our Vnet demo on [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans)
You can run the demo in our [Aistudio project](https://aistudio.baidu.com/aistudio/projectdetail/3519594) as well or run locally with the following steps.
- Download our repository.
    ```
    git clone https://github.com/PaddleCV-SIG/MedicalSeg.git
    cd MedicalSeg/
    ```
- Install requirements:
    ```
    pip install -r requirements.txt
    ```
- (Optional) Install CuPY if you want to accelerate the preprocess process. [CuPY installation guide](https://docs.cupy.dev/en/latest/install.html)

- Get and preprocess the data:
    - change the GPU setting [here](tools/preprocess_globals.yml) to True if you installed CuPY and want to use GPU to accelerate.
    ```
    python tools/prepare_lung_coronavirus.py
    ```

- Run the train and validation example. (Refer to the following usage to get the correct result.)
   ```
   sh run-vnet.sh
   ```

## 2. Get to Know our project
This part shows you the whole picture of our repo and details about the whole training and inference process. Our file tree is as follows:

```bash
├── configs         # All configuration stays here. If you use our model, you only need to change this and run-vnet.sh.
├── data            # Data stays here.
├── deploy          # deploy related doc and script.
├── medicalseg  
│   ├── core        # the core training, val and test file.
│   ├── datasets  
│   ├── models  
│   ├── transforms  # the online data transforms
│   └── utils       # all kinds of utility files
├── export.py
├── run-unet.sh     # the script to reproduce our project, including training, validate, infer and deploy
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
├── val.py
└── visualize.ipynb # You can try to visualize the result use this file.
```

### 2.1 Set configuration
Change configuration about loss, optimizer, dataset, and so on here. Our configurations is organized as follows:
```bash
├── _base_                   # base config, set your data path here and make sure you have enough space under this path.
│   └── global_configs.yml
├── lung_coronavirus         # each dataset has one config directory.
│   ├── lung_coronavirus.yml # all the config besides model is here, you can change configs about loss, optimizer, dataset, and so on.
│   ├── README.md  
│   ├── unet3d_lung_coronavirus_128_128_128_15k.yml  
│   └── vnet_lung_coronavirus_128_128_128_15k.yml    # model related config is here
└── schedulers              # the two stage scheduler, we have not use this part yet
    └── two_stage_coarseseg_fineseg.yml
```


### 2.2 Prepare the data
We use the data preparation script to download, preprocess, convert, and split the data automatically. If you want to prepare the data as we did, you can run the data prepare file like the following:
```
python tools/prepare_lung_coronavirus.py # take the CONVID-19 CT scans as example.
```

### 2.3 Train & Validate

After changing your config, you are ready to train your model. A basic training and validation example is [run-vnet.sh](./run-vnet.sh). Let's see some of the training and validation configurations in this file.

```bash
# set your GPU ID here
export CUDA_VISIBLE_DEVICES=0

# set the config file name and save directory here
yml=vnet_lung_coronavirus_128_128_128_15k
save_dir=saved_model/${yml}
mkdir save_dir

# Train the model: see the train.py for detailed explanation on script args
python3 train.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# Validate the model: see the val.py for detailed explanation on script args
python3 val.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams

```


### 2.4 deploy the model

With a trained model, we support deploying it with paddle inference to boost the inference speed. The instruction to do so is as follows, and you can see a detailed tutorial [here](./deploy/python/README.md).

```bash
cd MedicalSeg/

# Export the model with trained parameter
python export.py --config configs/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k.yml --model_path /path/to/your/trained/model

# Infer it with Paddle Inference Python API
python deploy/python/infer.py \
    --config /path/to/model/deploy.yaml \
    --image_path /path/to/image/path/or/dir/
    --benchmark True   # Use it after installed AutoLog, to record the speed, see ./deploy/python/README.md for detail to install AutoLog.

```
If you see the "finish" output, you have sucessfully upgrade your model's infer speed.

## 3. Train on your own dataset
If you want to train on your dataset, simply add a [dataset file](./medicalseg/datasets/lung_coronavirus.py), a [data preprocess file](./tools/prepare_lung_coronavirus.py), a [configuration directory](./configs/lung_coronavirus), a [training](run-vnet.sh) script and you are good to go. Details on how to add can refer to the links above.

### 3.1. Add a configuration directory
As we mentioned, every dataset has its own configuration directory. If you want to add a new dataset, you can replicate the lung_coronavirus directory and change relevant names and configs.
```
├── _base_
│   └── global_configs.yml
├── lung_coronavirus
│   ├── lung_coronavirus.yml
│   ├── README.md
│   ├── unet3d_lung_coronavirus_128_128_128_15k.yml
│   └── vnet_lung_coronavirus_128_128_128_15k.yml
```

### 3.2. Add a new data preprocess file
Your data needs to be convert into numpy array and split into trainset and valset as our format. You can refer to the [prepare script](./tools/prepare_lung_coronavirus.py):

```python
├── lung_coronavirus_phase0  # the preprocessed file
│   ├── images
│   │   ├── imagexx.npy
│   │   ├── ...
│   ├── labels
│   │   ├── labelxx.npy
│   │   ├── ...
│   ├── train_list.txt       # put all train data names here, each line contains:  //path/to/img_name_xxx.npy /path/to/label_names_xxx.npy
│   └── val_list.txt         # put all val data names here, each line contains:  img_name_xxx.npy label_names_xxx.npy
```

### 3.3. Add a dataset file
Our dataset file inherits MedicalDataset base class, where data split is based on the train_list.txt and val_list.txt you generated from previous step. For more details, please refer to the [dataset script](./medicalseg/datasets/lung_coronavirus.py).

### 3.4. Add a run script
The run script is used to automate a series of process. To add your config file, just replicate the [run-vnet.sh](run-vnet.sh) and change it based on your thought. Here is the content of what they mean:
```bash
# set your GPU ID here
export CUDA_VISIBLE_DEVICES=0

# set the config file name and save directory here
yml=lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k  # relative path to your yml from config dir
config_name = vnet_lung_coronavirus_128_128_128_15k         # name of the config yml
save_dir_all=saved_model                                    # overall save dir
save_dir=saved_model/${config_name}                         # savedir of this exp
```

## 4. Acknowledgements
Many thanks to [Lin Han](https://github.com/linhandev), [Lang Du](https://github.com/justld), [onecatcn](https://github.com/onecatcn) for their contribution in  our repository, and to [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) for their powerful visualization toolkit that we used to present our visualizations.
