#!/bin/bash

set -o errexit
set -o nounset
shopt -s extglob

FILENAME=$1
# MODE be 'js_infer'
MODE=$2
# js_infer MODE , load model file and convert model to js_infer
if [ ${MODE} != "js_infer" ];then
    echo "Please change mode to 'js_infer'"
    exit
fi

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

# model_name
model_name=$(func_parser_value "${lines[1]}")
# model_path
model_path=tests/web/models/

rm -rf $model_path
echo ${model_path}${model_name}

# download inference model and  export to static shape model
wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz
cd $model_path && tar xf ppseg_lite_portrait_398x224.tar.gz && cd ../../../
python3 export.py \
    --config configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml \
    --model_path $model_path$model_name/model.pdparams \
    --save_dir $model_path$model_name/ \
    --input_shape 1 3 224 398 \
    --without_argmax --with_softmax

# convert inference model to web model: model.json、chunk_1.dat
cd tests/converter && python3 convertToPaddleJSModel.py \
    --modelPath=../../$model_path/$model_name/model.pdmodel \
    --paramPath=../../$model_path/$model_name/model.pdiparams \
    --outputDir=../../$model_path/$model_name/


# prepare paddle.js env
source_code_path=$(func_parser_value "${lines[3]}")
echo ${source_code_path}


cd ../ && npm install @paddlejs-models/humanseg@latest jest jest-puppeteer puppeteer
rm package-lock.json

