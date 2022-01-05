#!/bin/bash

set -o errexit
set -o nounset
shopt -s extglob

# paddlejs prepare 主要流程
# 1. 判断 node, npm 是否安装
# 2. 下载测试模型，当前为 ppseg_lite_portrait_398x224 ，如果需要替换，把对应模型连接和模型包名字修改即可
# - 当前模型：https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz
# - 模型配置：configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml
# 3. 导出模型为静态图模型
# 4. 转换模型， model.pdmodel model.pdiparams 转换为 model.json chunk.dat
# 5. 安装最新版本 humanseg sdk  @paddlejs-models/humanseg@latest
# 6. 安装测试环境依赖 puppeteer、jest、jest-puppeteer

# 判断是否安装了node
if ! type node >/dev/null 2>&1; then
    echo -e "\033[31m node 未安装 \033[0m"
    exit
fi

# 判断是否安装了npm
if ! type npm >/dev/null 2>&1; then
    echo -e "\033[31m npm 未安装 \033[0m"
    exit
fi

# MODE be 'js_infer'
MODE=$1
# js_infer MODE , load model file and convert model to js_infer
if [ ${MODE} != "js_infer" ];then
    echo "Please change mode to 'js_infer'"
    exit
fi


# saved_model_name
saved_model_name=pphumanseg_lite
# model_path
model_path=test_tipc/web/models/

rm -rf $model_path
echo ${model_path}${saved_model_name}

# download inference model
wget -nc -P $model_path https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz
cd $model_path && tar xf ppseg_lite_portrait_398x224.tar.gz && cd ../../../

# export to static shape model
python3 export.py \
    --config configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml \
    --model_path test_tipc/web/models/ppseg_lite_portrait_398x224/model.pdparams \
    --save_dir $model_path$saved_model_name/ \
    --input_shape 1 3 224 398 \
    --without_argmax --with_softmax


pip3 install paddlejsconverter
# convert inference model to web model: model.json、chunk_1.dat
paddlejsconverter \
    --modelPath=$model_path$saved_model_name/model.pdmodel \
    --paramPath=$model_path$saved_model_name/model.pdiparams \
    --outputDir=$model_path$saved_model_name/ \




# always install latest humanseg sdk
cd test_tipc/web
echo -e "\033[33m Installing the latest humanseg sdk... \033[0m"
npm install @paddlejs-models/humanseg@latest
echo -e "\033[32m The latest humanseg sdk installed completely.!~ \033[0m"

# install dependencies
if [ `npm list --dept 0 | grep puppeteer | wc -l` -ne 0 ] && [ `npm list --dept 0 | grep jest | wc -l` -ne 0 ];then
    echo -e "\033[32m Dependencies have installed \033[0m"
else
    echo -e "\033[33m Installing dependencies ... \033[0m"
    npm install jest jest-puppeteer puppeteer
    echo -e "\033[32m Dependencies installed completely.!~ \033[0m"
fi

# del package-lock.json
rm package-lock.json