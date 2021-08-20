#!/bin/bash

set -o errexit
set -o nounset

FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2
if [ ${MODE} = "infer" ];then
    model_path=tests/output/norm_gpus_0_autocast_null/
    rm -rf $model_path
    wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip
    cd $model_path && unzip fcn_hrnetw18_small_v1_humanseg_192x192.zip && mv fcn_hrnetw18_small_v1_humanseg_192x192/model.pdparams . && cd ../../../
fi
rm -rf ./tests/data/
wget -nc -P ./tests/data/ https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
cd ./tests/data/ && unzip mini_supervisely.zip && cd ../../
