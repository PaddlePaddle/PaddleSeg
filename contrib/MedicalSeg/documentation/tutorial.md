English ｜ [简体中文](tutorial_cn.md)

This documentation shows the details on how to use our repository from setting configurations to deploy.

## 1. Set configuration
Change configuration about loss, optimizer, dataset, and so on here. Our configurations is organized as follows:
```bash
├── _base_                   # base config, set your data path here and make sure you have enough space under this path.
│   └── global_configs.yml
├── lung_coronavirus         # each dataset has one config directory.
│   ├── lung_coronavirus.yml # all the config besides model is here, you can change configs about loss, optimizer, dataset, and so on.
│   ├── README.md  
│   └── vnet_lung_coronavirus_128_128_128_15k.yml    # model related config is here
└── schedulers              # the two stage scheduler, we have not use this part yet
    └── two_stage_coarseseg_fineseg.yml
```


## 2. Prepare the data
We use the data preparation script to download, preprocess, convert, and split the data automatically. If you want to prepare the data as we did, you can run the data prepare file like the following:
```
python tools/prepare_lung_coronavirus.py # take the CONVID-19 CT scans as example.
```

## 3. Train & Validate

After changing your config, you are ready to train your model. A basic training and validation example is [run-vnet.sh](../run-vnet.sh). Let's see some of the training and validation configurations in this file.

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


## 4. deploy the model

With a trained model, we support deploying it with paddle inference to boost the inference speed. The instruction to do so is as follows, and you can see a detailed tutorial [here](../deploy/python/README.md).

```bash
cd MedicalSeg/

# Export the model with trained parameter
python export.py --config configs/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k.yml --model_path /path/to/your/trained/model

# Infer it with Paddle Inference Python API
python deploy/python/infer.py \
    --config /path/to/model/deploy.yaml \
    --image_path /path/to/image/path/or/dir/
    --benchmark True   # Use it after installed AutoLog, to record the speed, see ../deploy/python/README.md for detail to install AutoLog.

```
If you see the "finish" output, you have sucessfully upgrade your model's infer speed.

## 5. Train on your own dataset
If you want to train on your dataset, simply add a [dataset file](../medicalseg/datasets/lung_coronavirus.py), a [data preprocess file](../tools/prepare_lung_coronavirus.py), a [configuration directory](../configs/lung_coronavirus), a [training](run-vnet.sh) script and you are good to go. Details on how to add can refer to the links above.

### 5.1 Add a configuration directory
As we mentioned, every dataset has its own configuration directory. If you want to add a new dataset, you can replicate the lung_coronavirus directory and change relevant names and configs.
```
├── _base_
│   └── global_configs.yml
├── lung_coronavirus
│   ├── lung_coronavirus.yml
│   ├── README.md
│   └── vnet_lung_coronavirus_128_128_128_15k.yml
```

### 5.2 Add a new data preprocess file
Your data needs to be convert into numpy array and split into trainset and valset as our format. You can refer to the [prepare script](../tools/prepare_lung_coronavirus.py):

```python
├── lung_coronavirus_phase0  # the preprocessed file
│   ├── images
│   │   ├── imagexx.npy
│   │   ├── ...
│   ├── labels
│   │   ├── labelxx.npy
│   │   ├── ...
│   ├── train_list.txt       # put all train data names here, each line contains:  /path/to/img_name_xxx.npy /path/to/label_names_xxx.npy
│   └── val_list.txt         # put all val data names here, each line contains:  img_name_xxx.npy label_names_xxx.npy
```

### 5.3 Add a dataset file
Our dataset file inherits MedicalDataset base class, where data split is based on the train_list.txt and val_list.txt you generated from previous step. For more details, please refer to the [dataset script](../medicalseg/datasets/lung_coronavirus.py).

### 5.4 Add a run script
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
