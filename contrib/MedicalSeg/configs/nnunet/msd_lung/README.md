# [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
针对96个3D肺部肿瘤数据进行训练 (包含 64 例训练 + 32 例测试)

## 性能

### NNUnet
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
| 主干网络 | 分辨率 | 学习率 | 训练轮数 | Dice |  链接 |
|:-:|:-:|:-:|:-:|:-:|:-:|
|2D|512x512|0.01|30000|53.549%|[model_fold0](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/2d_fold0/model.pdparams) \| [model_fold1](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/2d_fold1/model.pdparams) \| [model_fold2](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/2d_fold2/model.pdparams) \| [model_fold3](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/2d_fold3/model.pdparams) \| [model_fold4](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/2d_fold4/model.pdparams) \|  [log](https://aistudio.baidu.com/aistudio/datasetdetail/150774)|
|3D lowres|80x192x160|0.01|30000|68.281%|[model_fold0](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/3dlowres_fold0/model.pdparams) \| [model_fold1](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/3dlowres_fold1/model.pdparams) \| [model_fold2](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/3dlowres_fold2/model.pdparams) \| [model_fold3](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/3dlowres_fold3/model.pdparams) \| [model_fold4](https://bj.bcebos.com/paddleseg/paddleseg3d/msd_lung/3dlowres_fold4/model.pdparams)   \| [log](https://aistudio.baidu.com/aistudio/datasetdetail/150774)|
|3D fullres|80x192x160|0.01|30000|66.281% |[model](https://aistudio.baidu.com/aistudio/datasetdetail/162872)  \| [log](https://aistudio.baidu.com/aistudio/datasetdetail/150774)|
|3D cascade|80x192x160|0.01|40000|67.996%|[model](https://aistudio.baidu.com/aistudio/datasetdetail/163284) \| [log](https://aistudio.baidu.com/aistudio/datasetdetail/150774)|

# 使用示例  
AiStudio使用示例：[MedicalSeg-nnUNet使用教程](https://aistudio.baidu.com/aistudio/projectdetail/4884907?contributionType=1)

# 简介
nnUNet包含2D-UNet，3d-UNet，Cascade UNet共3个模型，每个模型使用五折交叉验证的方式训练，由于Cascade UNet包含low resolution和high resolution 2个模型，故共有20个模型，对应20个配置文件。

# 数据准备
本教程以[MSD Lung](http://medicaldecathlon.com/)数据集为例，如果使用其他数据集，仅需要修改配置文件中的数据集路径和plan路径即可。解压数据集，目录结构为
```
 MeidicalSeg
 |-- msd_lung
     |-- Task06_Lung
        |-- imagesTr
            |-- lung_001.nii.gz
            |-- lung_003.nii.gz
        |-- ImagesTs
            |-- lung_002.nii.gz
        |-- labelsTr
            |-- lung_001.nii.gz
        |-- dataset.json
```

# 训练
训练命令和其他算法大部分相同，训练nnUNet是需要添加--nnunet，为了降低模型使用的显存和训练速度，使用混合精度训练。此处先介绍2d-UNet和3d-UNet的训练-验证-预测流程，Cascade UNet流程后续稍微有些区别，后续单独讲述。

训练命令如下，需要注意以下几点：  
1、如果在训练阶段开启验证(命令中包含--do_eval)，因为使用了和nnUNet一样的验证集采样策略，验证的精度不可信，最好不要使用best_model文件夹下的权重；  
2、配置文件中的参数，除了数据集路径相关的参数，其他参数若非您知道其含义，否则请勿随意更改。  
```commandline
python train.py --config {config path} --log_iters 20 --precision fp16 --nnunet --save_dir {output dir} --save_interval 1000 --use_vdl
```

2d-UNet 训练命令：
```commandline
python train.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold0.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/2d_unet/fold0 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold1.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/2d_unet/fold1 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold2.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/2d_unet/fold2 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold3.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/2d_unet/fold3 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold4.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/2d_unet/fold4 --save_interval 1000 --use_vdl
```

3d-UNet训练命令：
```commandline
python train.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold0.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/3d_unet/fold0 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold1.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/3d_unet/fold1 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold2.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/3d_unet/fold2 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold3.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/3d_unet/fold3 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold4.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/3d_unet/fold4 --save_interval 1000 --use_vdl
```

# 验证
五折交叉验证的方式，需要对每一折的验证集单独验证，让后将五个验证集合并起来检查其精度，验证集预测结果储存在val_save_folder指定的目录下。验证结束后，目录结构为：
```
 MeidicalSeg
 |-- output
     |-- 2d_val
        |-- cv_niftis_postprocessed
            |-- lung_001.nii.gz
            |-- lung_003.nii.gz
        |-- cv_niftis_raw
            |-- lung_001.nii.gz
            |-- lung_003.nii.gz
        |-- fold_0
            |-- cv_niftis_raw
            |-- cv_niftis_postprocessed
        |-- fold_1
            |-- cv_niftis_raw
            |-- cv_niftis_postprocessed
        |-- fold_2
            |-- cv_niftis_raw
            |-- cv_niftis_postprocessed
        |-- fold_3
            |-- cv_niftis_raw
            |-- cv_niftis_postprocessed
        |-- fold_4
            |-- cv_niftis_raw
            |-- cv_niftis_postprocessed
        |-- gt_niftis
        |-- postprocessing.json
```
fold_0至fold_4为五折单独验证结果，cv_niftis_raw是未进行后处理的预测结果，cv_niftis_postprocessed是进行后处理后的预测结果，gt_niftis中保存数据集的标签，postprocessing.json保存着验证结果。

首先进行单折验证，命令如下：
```commandline
python nnunet/single_fold_eval.py --config {config path} --model_path {model path} --val_save_folder {val output folder} --precision fp16
```

五折单独验证完后，进行整体验证，命令如下(val_save_folder文件夹内的gt_niftis就是GT，该目录会在单折验证时自动生存)：
```commandline
python nnunet/all_folds_eval.py --gt_dir {val output folder}/gt_niftis --val_pred_dir {val output folder}
```

2D-UNet验证命令（模型路径需要更改为自己的模型路径）：
```commandline
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold0.yml --model_path output/2d_unet/fold0/iter_30000/model.pdparams --val_save_folder output/2d_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold1.yml --model_path output/2d_unet/fold1/iter_30000/model.pdparams --val_save_folder output/2d_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold2.yml --model_path output/2d_unet/fold2/iter_30000/model.pdparams --val_save_folder output/2d_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold3.yml --model_path output/2d_unet/fold3/iter_30000/model.pdparams --val_save_folder output/2d_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold4.yml --model_path output/2d_unet/fold4/iter_30000/model.pdparams --val_save_folder output/2d_val --precision fp16

python nnunet/all_folds_eval.py --gt_dir output/2d_val/gt_niftis --val_pred_dir output/2d_val
```

3D-UNet验证命令（模型路径需要更改为自己的模型路径）：
```commandline
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold0.yml --model_path output/3d_unet/fold0/iter_30000/model.pdparams --val_save_folder output/3dfullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold1.yml --model_path output/3d_unet/fold1/iter_30000/model.pdparams --val_save_folder output/3dfullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold2.yml --model_path output/3d_unet/fold2/iter_30000/model.pdparams --val_save_folder output/3dfullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold3.yml --model_path output/3d_unet/fold3/iter_30000/model.pdparams --val_save_folder output/3dfullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold4.yml --model_path output/3d_unet/fold4/iter_30000/model.pdparams --val_save_folder output/3dfullres_val --precision fp16

python nnunet/all_folds_eval.py --gt_dir output/3dfullres_val/gt_niftis --val_pred_dir output/3dfullres_val
```

# Cascade UNet
Cascade UNet使用方法和上述方法类似，但是Cascade UNet第二阶段的模型（fullres）的输入中包含第一阶段的模型(lowres)预测结果，故使用起来有些许区别。

第一步：训练lowres
```commandline
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold0.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold0 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold1.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold1 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold2.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold2 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold3.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold3 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold4.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold4 --save_interval 1000 --use_vdl
```

第二步： 验证lowres（加上--predict_next_stage得到下一阶段的输入）
```commandline
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold0.yml --model_path output/cascade_lowres/fold0/iter_30000/model.pdparams --val_save_folder output/cascade_lowres_val --precision fp16 --predict_next_stage
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold1.yml --model_path output/cascade_lowres/fold1/iter_30000/model.pdparams --val_save_folder output/cascade_lowres_val --precision fp16 --predict_next_stage
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold2.yml --model_path output/cascade_lowres/fold2/iter_30000/model.pdparams --val_save_folder output/cascade_lowres_val --precision fp16 --predict_next_stage
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold3.yml --model_path output/cascade_lowres/fold3/iter_30000/model.pdparams --val_save_folder output/cascade_lowres_val --precision fp16 --predict_next_stage
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold4.yml --model_path output/cascade_lowres/fold4/iter_30000/model.pdparams --val_save_folder output/cascade_lowres_val --precision fp16 --predict_next_stage
```

第三步： 训练fullres
```commandline
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold0.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_fullres/fold0 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold1.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_fullres/fold1 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold2.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_fullres/fold2 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold3.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_fullres/fold3 --save_interval 1000 --use_vdl
python train.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold4.yml --log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_fullres/fold4 --save_interval 1000 --use_vdl
```

第四步： 验证fullres
```commandline
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold0.yml --model_path output/cascade_fullres/fold0/iter_30000/model.pdparams --val_save_folder output/cascade_fullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold1.yml --model_path output/cascade_fullres/fold1/iter_30000/model.pdparams --val_save_folder output/cascade_fullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold2.yml --model_path output/cascade_fullres/fold2/iter_30000/model.pdparams --val_save_folder output/cascade_fullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold3.yml --model_path output/cascade_fullres/fold3/iter_30000/model.pdparams --val_save_folder output/cascade_fullres_val --precision fp16
python nnunet/single_fold_eval.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold4.yml --model_path output/cascade_fullres/fold4/iter_30000/model.pdparams --val_save_folder output/cascade_fullres_val --precision fp16
```

# Ensemble
经过以上步骤，得到了2d-UNet,3d-UNet,Cascade UNet的各自验证结果，保存的文件夹分别为：output/2d_val,output/3dfullres_val,output/cascade_fullres_val,下面是进行模型集成的命令.  
说明：  
1、plan_path可以选择plan2D路径或者plan3D路径皆可，需要用到其中包含的类别信息；  
2、gt_dir指定任意验证目录下的gt_niftis即可；  
3、模型集成会在输出目录下的postprocessing.json中看到集成后的精度，可以选择集成后精度最高的模型来集成预测。  
```commandline
python nnunet/ensemble.py --ensemble_folds output/3dfullres_val output/cascade_lowres_val output/2d_val --gt_dir output/cascade_lowres_val/gt_niftis --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_2D.pkl --output_folder output/ensemble
```

# Predict
在预测阶段，需要动态加载五折对应的五个权重，故仅支持动态图预测。如果要集成预测结果，需要分别先预测各个模型的预测结果，然后再集成。  
说明：  
1、如果预测结果要ensemble，加上--save_npz参数；  
2、Cascade UNet预测时，先预测lowres结果，再预测fullres；  
3、plan_path选择对应的模型plan;  
4、postprocessing_json_path对应该模型验证目录下的postprocessing.json；  
5、model_type 支持2d 3d cascade_lowres cascade_lowres，分别对应2D-UNet 3D-UNet Cascade UNet lowres Cascade UNet fullres.  
6、如果不需要后处理，加上--disable_postprocessing参数（当验证时发现某个模型不使用后处理的精度高，可不使用后处理）；

预测命令：  
```commandline
python nnunet/predict.py --image_folder {image folder}  --output_folder {output folder} --plan_path {plan path} --model_paths {model path0} {model path1} {...} --postprocessing_json_path {postprocessing.json path} --model_type 3d --disable_postprocessing  --save_npz
```

2D-UNet预测命令:
```commandline
python nnunet/predict.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_predict/2d_unet --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_2D.pkl --model_paths output/2d_unet/fold0/iter_30000/model.pdparams output/2d_unet/fold1/iter_30000/model.pdparams output/2d_unet/fold2/iter_30000/model.pdparams output/2d_unet/fold3/iter_30000/model.pdparams output/2d_unet/fold4/iter_30000/model.pdparams --postprocessing_json_path output/2d_val/postprocessing.json --model_type 2d --save_npz
```

3D-UNet预测命令:
```commandline
python nnunet/predict.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_predict/3d_unet --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/3d_unet/fold0/iter_30000/model.pdparams output/3d_unet/fold1/iter_30000/model.pdparams output/3d_unet/fold2/iter_30000/model.pdparams output/3d_unet/fold3/iter_30000/model.pdparams output/3d_unet/fold4/iter_30000/model.pdparams --postprocessing_json_path output/3d_unet/postprocessing.json --model_type 3d --save_npz
```

Cascade UNet预测命令(cascade fullres需要cascade lowres的预测结果作为输入):
```commandline
python nnunet/predict.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_predict/lowres_pred --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/cascade_lowres/fold0/iter_30000/model.pdparams output/cascade_lowres/fold1/iter_30000/model.pdparams output/cascade_lowres/fold2/iter_30000/model.pdparams output/cascade_lowres/fold3/iter_30000/model.pdparams output/cascade_lowres/fold4/iter_30000/model.pdparams --postprocessing_json_path output/cascade_lowres_val/postprocessing.json --model_type cascade_lowres
python nnunet/predict.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_predict/fullres_pred --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/cascade_fullres/fold0/iter_30000/model.pdparams output/cascade_fullres/fold1/iter_30000/model.pdparams output/cascade_fullres/fold2/iter_30000/model.pdparams output/cascade_fullres/fold3/iter_30000/model.pdparams output/cascade_fullres/fold4/iter_30000/model.pdparams --postprocessing_json_path output/cascade_lowres_val/postprocessing.json --model_type cascade_fullres --save_npz --lowres_segmentations output/nnunet_predict/lowres_pred
```

现在得到了2D-UNet，3D-UNet，Cascade-UNet的预测结果，预测结果的ensemble命令如下：  
说明：  
1、ensemble_folds传入的文件夹是需要ensemble的预测结果目录；  
2、postprocessing_json_path可传入任意模型的postprocessing.json路径，该文件在验证目录下，具体参考上方的验证小节；
```commandline
python nnunet/ensemble.py --ensemble_folds {predict folder1} {predict folder2}  --output_folder {output folder} --postprocessing_json_path {postprocessing.json path}
```
以Cascade UNet和3D UNet为例：  
```commandline
python nnunet/ensemble.py --ensemble_folds output/nnunet_predict/fullres_pred output/nnunet_predict/3d_unet  --output_folder output/ensemble_pred --postprocessing_json_path output/cascade_lowres_val/postprocessing.json
```

# 模型导出
模型导出使用如下命令：
```commandline
python nnunet/export.py --config {config path} --save_dir {output dir} --model_path {path to pdparams}
```

2D-UNet导出：
```commandline
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold0.yml --save_dir output/static/2d_unet/fold0 --model_path output/2d_unet/fold0/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold1.yml --save_dir output/static/2d_unet/fold1 --model_path output/2d_unet/fold1/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold2.yml --save_dir output/static/2d_unet/fold2 --model_path output/2d_unet/fold2/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold3.yml --save_dir output/static/2d_unet/fold3 --model_path output/2d_unet/fold3/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_2d_msd_lung_fold4.yml --save_dir output/static/2d_unet/fold4 --model_path output/2d_unet/fold4/iter_30000/model.pdparams
```

3D-UNet导出：
```commandline
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold0.yml --save_dir output/static/3d_unet/fold0 --model_path output/3d_unet/fold0/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold1.yml --save_dir output/static/3d_unet/fold1 --model_path output/3d_unet/fold1/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold2.yml --save_dir output/static/3d_unet/fold2 --model_path output/3d_unet/fold2/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold3.yml --save_dir output/static/3d_unet/fold3 --model_path output/3d_unet/fold3/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3d_fullres_msd_lung_fold4.yml --save_dir output/static/3d_unet/fold4 --model_path output/3d_unet/fold4/iter_30000/model.pdparams
```

Cascade lowres导出：
```commandline
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold0.yml --save_dir output/static/cascade_lowres/fold0 --model_path output/cascade_lowres/fold0/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold1.yml --save_dir output/static/cascade_lowres/fold1 --model_path output/cascade_lowres/fold1/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold2.yml --save_dir output/static/cascade_lowres/fold2 --model_path output/cascade_lowres/fold2/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold3.yml --save_dir output/static/cascade_lowres/fold3 --model_path output/cascade_lowres/fold3/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_lowres_msd_lung_fold4.yml --save_dir output/static/cascade_lowres/fold4 --model_path output/cascade_lowres/fold4/iter_30000/model.pdparams
```

Cascade fullres导出：
```commandline
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold0.yml --save_dir output/static/cascade_fullres/fold0 --model_path output/cascade_fullres/fold0/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold1.yml --save_dir output/static/cascade_fullres/fold1 --model_path output/cascade_fullres/fold1/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold2.yml --save_dir output/static/cascade_fullres/fold2 --model_path output/cascade_fullres/fold2/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold3.yml --save_dir output/static/cascade_fullres/fold3 --model_path output/cascade_fullres/fold3/iter_30000/model.pdparams
python nnunet/export.py --config configs/nnunet/msd_lung/nnunet_3dcascade_fullres_msd_lung_fold4.yml --save_dir output/static/cascade_fullres/fold4 --model_path output/cascade_fullres/fold4/iter_30000/model.pdparams
```

# 静态图推理
推理命令和预测命令非常相似，区别在于将pdparams路径更换为pdmodel和pdiparams路径。命令为：
```commandline
python nnunet/infer.py --image_folder {image dir}  --output_folder {output dir} --plan_path {plan path} --model_paths {pdmodel path1} {pdmodel path2} {...} --param_paths {pdiparams path1} {pdiparams path2} {...} --postprocessing_json_path {postprocessing json path} --model_type 3d --disable_postprocessing  --save_npz
```

2D-UNet推理：
```commandline
python nnunet/infer.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_static/2d_unet --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/static/2d_unet/fold0/model.pdmodel output/static/2d_unet/fold1/model.pdmodel output/static/2d_unet/fold2/model.pdmodel output/static/2d_unet/fold3/model.pdmodel output/static/2d_unet/fold4/model.pdmodel --param_paths output/static/2d_unet/fold0/model.pdiparams output/static/2d_unet/fold1/model.pdiparams output/static/2d_unet/fold2/model.pdiparams output/static/2d_unet/fold3/model.pdiparams output/static/2d_unet/fold4/model.pdiparams --postprocessing_json_path output/2d_unet_val/postprocessing.json --model_type 3d --disable_postprocessing  --save_npz
```

3D-UNet推理：
```commandline
python nnunet/infer.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_static/3d_fullres --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/static/3d_unet/fold0/model.pdmodel output/static/3d_unet/fold1/model.pdmodel output/static/3d_unet/fold2/model.pdmodel output/static/3d_unet/fold3/model.pdmodel output/static/3d_unet/fold4/model.pdmodel --param_paths output/static/3d_unet/fold0/model.pdiparams output/static/3d_unet/fold1/model.pdiparams output/static/3d_unet/fold2/model.pdiparams output/static/3d_unet/fold3/model.pdiparams output/static/3d_unet/fold4/model.pdiparams --postprocessing_json_path output/3d_unet_val/postprocessing.json --model_type 3d --disable_postprocessing  --save_npz
```

Cascade lowres推理：
```commandline
python nnunet/infer.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_static/lowres_pred --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/static/cascade_lowres/fold0/model.pdmodel output/static/cascade_lowres/fold1/model.pdmodel output/static/cascade_lowres/fold2/model.pdmodel output/static/cascade_lowres/fold3/model.pdmodel output/static/cascade_lowres/fold4/model.pdmodel --param_paths output/static/cascade_lowres/fold0/model.pdiparams output/static/cascade_lowres/fold1/model.pdiparams output/static/cascade_lowres/fold2/model.pdiparams output/static/cascade_lowres/fold3/model.pdiparams output/static/cascade_lowres/fold4/model.pdiparams --postprocessing_json_path output/cascade_lowres_val/postprocessing.json --model_type cascade_lowres
```

Cascade fullres推理：
```commandline
python nnunet/infer.py --image_folder msd_lung/Task006_Lung/imagesTs  --output_folder output/nnunet_static/fullres_pred --plan_path msd_lung/preprocessed/Task006_Lung/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/static/cascade_fullres/fold0/model.pdmodel output/static/cascade_fullres/fold1/model.pdmodel output/static/cascade_fullres/fold2/model.pdmodel output/static/cascade_fullres/fold3/model.pdmodel output/static/cascade_fullres/fold4/model.pdmodel --param_paths output/static/cascade_fullres/fold0/model.pdiparams output/static/cascade_fullres/fold1/model.pdiparams output/static/cascade_fullres/fold2/model.pdiparams output/static/cascade_fullres/fold3/model.pdiparams output/static/cascade_fullres/fold4/model.pdiparams --postprocessing_json_path output/cascade_lowres_val/postprocessing.json --model_type cascade_fullres --save_npz --lowres_segmentations output/nnunet_static/lowres_pred
```
