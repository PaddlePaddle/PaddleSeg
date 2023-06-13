#!/usr/bin/env bash

source test_tipc/common_func.sh

set -o errexit
set -o nounset

FILENAME=$1
# $MODE must be one of ('lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer')
MODE=$2

dataline=$(cat ${FILENAME})

# Parse params
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")

# Download pretrained weights
if [ ${MODE} = 'whole_infer' ]; then
    :
fi

# Download datasets
DATA_DIR='./test_tipc/data/'
mkdir -p "${DATA_DIR}"
if [[ ${MODE} = 'lite_train_lite_infer' \
    || ${MODE} = 'lite_train_whole_infer' \
    || ${MODE} = 'whole_infer' ]]; then

    download_and_unzip_dataset "${DATA_DIR}" coco https://paddleseg.bj.bcebos.com/dataset/mini_coco_panseg.zip

elif [[ ${MODE} = 'whole_train_whole_infer' ]]; then

    # Currently, 'whole_train_whole_infer' mode is not supported.
    :

fi