#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# disable mkldnn on non x86_64 env
arch=$(uname -i)
if [ $arch != "x86_64" ]; then
    sed -i "s/--enable_mkldnn:True|False/--enable_mkldnn:False/g" $FILENAME
    sed -i "s/--enable_mkldnn:True/--enable_mkldnn:False/g" $FILENAME
fi

# change gpu to npu in tipc txt configs
sed -i "s/Global.use_gpu/Global.use_npu/g" $FILENAME
sed -i "s/--device gpu/--device npu/g" $FILENAME
sed -i "s/--device:gpu/--device:npu/g" $FILENAME
sed -i "s/--device:cpu|gpu/--device:cpu|npu/g" $FILENAME
# disable benchmark as AutoLog required nvidia-smi command
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
# python has been updated to version 3.9 for npu backend
sed -i "s/python3.7/python3.9/g" $FILENAME
dataline=`cat $FILENAME`

# parser params
IFS=$'\n'
lines=(${dataline})
modelname=$(func_parser_value "${lines[1]}")

if  [ $modelname == "hrnet_w48_contrast" ] || [ $modelname == "pfpnnet" ] || [ $modelname == "knet" ] \
 || [ $modelname == "maskformer" ] || [ $modelname == "ocrnet_hrformer_small" ];then
    sed -i "s/lite_train_lite_infer=20/lite_train_lite_infer=2/g" $FILENAME
    sed -i "s/--save_interval 500/--save_interval 50/g" $FILENAME
fi

# change gpu to npu in execution script
sed -i "s/\"gpu\"/\"npu\"/g" test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo -e "\033[1;32m Started to run command: ${cmd}!  \033[0m"
eval $cmd
