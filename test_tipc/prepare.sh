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

trainer_list=$(func_parser_value "${lines[14]}")

model_path=test_tipc/output/${model_name}/

# download pretrained model
if [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ]; then
    rm -rf $model_path
    if [ ${model_name} == "fcn_hrnetw18_small" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip
        cd $model_path && unzip fcn_hrnetw18_small_v1_humanseg_192x192.zip  &&  cd -
    elif [ ${model_name} == "pphumanseg_lite" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/pphumanseg_lite_generic_192x192.zip
        cd $model_path && unzip pphumanseg_lite_generic_192x192.zip  &&  cd -
    elif [ ${model_name} == "deeplabv3p_resnet50" ];then
        wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip
        cd $model_path && unzip deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip && cd -

    fi
fi
# download data
if [ ${model_name} == "fcn_hrnetw18_small" ] || [ ${model_name} == "pphumanseg_lite" ] || [ ${model_name} == "deeplabv3p_resnet50" ];then
    rm -rf ./test_tipc/data/mini_supervisely
    wget -nc -P ./test_tipc/data/ https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
    cd ./test_tipc/data/ && unzip mini_supervisely.zip && cd -
fi
