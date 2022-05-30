#!/bin/bash
source test_tipc/common_func.sh

set -o errexit
set -o nounset

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
# The training params
model_name=$(func_parser_value "${lines[1]}")

#trainer_list=$(func_parser_value "${lines[14]}")

model_path=test_tipc/output/${model_name}/


if [ ${MODE} = "serving_infer" ]; then
    inference_models=test_tipc/inferences/${model_name}/
    mkdir -p $inference_models
    cd $inference_models && rm -rf * && cd -

    if [ ${model_name} == "stdc_stdc1" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/dygraph/demo/stdc1seg_infer_model.tar.gz
        tar xf $inference_models/stdc1seg_infer_model.tar.gz -C $inference_models
    elif [ ${model_name} == "pp_liteseg_stdc1" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz
        tar xf $inference_models/pp_liteseg_infer_model.tar.gz  -C $inference_models
    elif [ ${model_name} == "pp_liteseg_stdc2" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.zip
        unzip $inference_models/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.zip -d $inference_models/
    elif [ ${model_name} == "pp_humanseg_lite" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/pp_humanseg_lite_export_398x224.zip
        unzip $inference_models/pp_humanseg_lite_export_398x224.zip -d $inference_models/
    elif [ ${model_name} == "pp_humanseg_mobile" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/pp_humanseg_mobile_export_192x192.zip
        unzip $inference_models/pp_humanseg_mobile_export_192x192.zip -d $inference_models/
    elif [ ${model_name} == "pp_humanseg_server" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/pp_humanseg_server_export_512x512.zip
        unzip $inference_models/pp_humanseg_server_export_512x512.zip -d $inference_models/
    elif [ ${model_name} == "fcn_hrnetw18" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/fcn_hrnetw18_cityscapes_1024x512_80k.zip
        unzip $inference_models/fcn_hrnetw18_cityscapes_1024x512_80k.zip -d $inference_models/
    elif [ ${model_name} == "ocrnet_hrnetw48" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/ocrnet_hrnetw48_cityscapes_1024x512_160k.zip
        unzip $inference_models/ocrnet_hrnetw48_cityscapes_1024x512_160k.zip -d $inference_models/
    elif [ ${model_name} == "ocrnet_hrnetw18" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/ocrnet_hrnetw18_cityscapes_1024x512_160k.zip
        unzip $inference_models/ocrnet_hrnetw18_cityscapes_1024x512_160k.zip -d $inference_models/
    elif [ ${model_name} == "pp_humanseg_matting" ];then
        wget -P $inference_models https://paddleseg.bj.bcebos.com/tipc/infer_models/pp-humanmatting-resnet34_vd.zip
        unzip $inference_models/pp-humanmatting-resnet34_vd.zip -d $inference_models/
    fi
fi

# download pretrained model
if [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ]; then
    if [ ${model_name} == "fcn_hrnetw18_small" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip
        cd $model_path && unzip fcn_hrnetw18_small_v1_humanseg_192x192.zip  &&  cd -
    elif [ ${model_name} == "pphumanseg_lite" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/pphumanseg_lite_generic_192x192.zip
        cd $model_path && unzip pphumanseg_lite_generic_192x192.zip  &&  cd -
    elif [ ${model_name} == "deeplabv3p_resnet50" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip
        cd $model_path && unzip deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip && cd -
    elif [ ${model_name} == "bisenetv2" ];then
        wget -nc -P $model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
    elif [ ${model_name} == "ocrnet_hrnetw18" ];then
        wget -nc -P $model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams
    elif [ ${model_name} == "segformer_b0" ];then
        wget -nc -P $model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b0_cityscapes_1024x1024_160k/model.pdparams
    elif [ ${model_name} == "stdc_stdc1" ];then
        wget -nc -P $model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/stdc1_seg_cityscapes_1024x512_80k/model.pdparams
    elif [ ${model_name} == "ppmatting" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/matting/models/modnet-mobilenetv2.pdparams
    elif [ ${model_name} == "pp_liteseg_stdc1" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/model.pdparams
    elif [ ${model_name} == "pp_liteseg_stdc2" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/model.pdparams
    elif [ ${model_name} == "ddrnet" ];then
        wget -nc -P $model_path https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ddrnet23_cityscapes_1024x1024_120k/model.pdparams
    fi
fi

# download data
if [ ${MODE} = "benchmark_train" ];then
    pip install -r requirements.txt
    mkdir -p ./test_tipc/data
    if [ ${model_name} == "deeplabv3p_resnet50" ] || [ ${model_name} == "fcn_hrnetw18" ] ;then   # 需要使用全量数据集,否则性能下降
        wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar -O ./test_tipc/data/cityscapes.tar
        tar -xf ./test_tipc/data/cityscapes.tar  -C ./test_tipc/data/
    else
        wget https://paddleseg.bj.bcebos.com/dataset/cityscapes_30imgs.tar.gz \
            -O ./test_tipc/data/cityscapes_30imgs.tar.gz
        tar -zxf ./test_tipc/data/cityscapes_30imgs.tar.gz -C ./test_tipc/data/
        mv ./test_tipc/data/cityscapes_30imgs ./test_tipc/data/cityscapes
    fi
elif [ ${MODE} == "serving_infer" ];then
    mkdir -p ./test_tipc/data
    wget -nc -P ./test_tipc/data https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
else
    if [ ${model_name} == "fcn_hrnetw18_small" ] || [ ${model_name} == "pphumanseg_lite" ] || [ ${model_name} == "deeplabv3p_resnet50" ];then
        rm -rf ./test_tipc/data/mini_supervisely
        wget -nc -P ./test_tipc/data/ https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
        cd ./test_tipc/data/ && unzip mini_supervisely.zip && cd -
    elif [ ${model_name} == "ppmatting" ];then
        rm -rf ./test_tipc/data/PPM-100
        wget -nc -P ./test_tipc/data/ https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip
        cd ./test_tipc/data/ && unzip PPM-100.zip && cd -
    else
        rm -rf ./test_tipc/data/cityscapes
        wget -nc -P ./test_tipc/data/ https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar
        cd ./test_tipc/data/ && tar -xf cityscapes.tar && cd -
    fi
fi

models=("enet" "bisenetv2" "ocrnet_hrnetw18" "ocrnet_hrnetw48" "deeplabv3p_resnet50_cityscapes" \
        "fastscnn" "fcn_hrnetw18" "pp_liteseg_stdc1" "pp_liteseg_stdc2" "ccnet")
if [ $(contains "${models[@]}" "${model_name}") == "y" ]; then
    cp ./test_tipc/data/cityscapes_val_5.list ./test_tipc/data/cityscapes
fi