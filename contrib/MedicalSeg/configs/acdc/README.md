# [Automated cardiac diagnosis](https://acdc.creatis.insa-lyon.fr/description/databases.html)
The database is made available to participants through two datasets from the dedicated online evaluation website after a personal registration: i) a training dataset of 100 patients along with the corresponding manual references based on the analysis of one clinical expert; ii) a testing dataset composed of 50 new patients, without manual annotations but with the patient information given above. The raw input images are provided through the Nifti format.
### Prepare dataset
To preprocess the ACDC data, you first need to download `training.zip` from https://acdc.creatis.insa-lyon.fr/#phase/5846c3ab6a3c7735e84b67f2
```
unzip training.zip
mkdir data/ACDCDataset
python tools/prepare_abdomen.py training/
```
The dataset will be automatically automatically preprocessed. The file structure is as follows:
```
ACDCDataset
|--clean_data
│   ├── labelsTr
│   │   ├──patient001_frame13_0000.nii.gz
│   │   ├──patient002_frame13_0000.nii.gz
│   │   ├──patient003_frame13_0000.nii.gz
│   │   │──........
│   │   ├──patient015_frame13_0000.nii.gz
│   ├── imagesTr
│   │   ├──patient001_frame13_0000.nii.gz
│   │   ├──patient002_frame13_0000.nii.gz
│   │   ├──patient003_frame13_0000.nii.gz
│   │   │──........
│   │   ├──patient015_frame13_0000.nii.gz
├── ACDCDataset_phase
│   ├── images
│   │   ├── patient030_frame12_0000.npy
│   │   └── ...
│   ├── labels
│   │   ├── patient030_frame12_0000.npy
│   │   └── ...
│   ├── train_list.txt
│   └── val_list.txt
```
Then you can start the training program, such as the following command:
```
python train.py --config configs/acdc/nnformer_acdc_160_160_14_250k.yml --save_interval 250 --num_workers 4 --do_eval --log_iters 250 --sw_num 1 --is_save_data False --has_dataset_json False
```

## Performance


### nnFormer
>   Hong-Yu Zhou, Student Member, IEEE, Jiansen Guo, Yinghao Zhang, Xiaoguang Han, Lequan Yu, Liansheng Wang, Member, IEEE, and Yizhou Yu, Fellow, IEEE

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|14x160x160|1e-4|250000|91.78%|[model](https://paddleseg.bj.bcebos.com/paddleseg3d/acdc/nnformer_acdc_160_160_14_250k_4e-4/model.pdparams)\| [log](https://paddleseg.bj.bcebos.com/paddleseg3d/acdc/nnformer_acdc_160_160_14_250k_4e-4/train.log)\| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=b9a90b8aba579997a6f088b840a6e96d)|
