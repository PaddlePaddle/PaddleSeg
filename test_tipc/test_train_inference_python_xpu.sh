#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# disable mkldnn on non x86_64 env
arch=$(uname -i)
if [ $arch != "x86_64" ]; then
    sed -i "s/--enable_mkldnn:True|False/--enable_mkldnn:False/g" $FILENAME
    sed -i "s/--enable_mkldnn:True/--enable_mkldnn:False/g" $FILENAME
fi

# change gpu to xpu in tipc txt configs
sed -i "s/Global.use_gpu/Global.use_xpu/g" $FILENAME
sed -i "s/--device gpu/--device xpu/g" $FILENAME
sed -i "s/--device:cpu|gpu/--device:cpu|xpu/g" $FILENAME
# disable benchmark as AutoLog required nvidia-smi command
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
dataline=`cat $FILENAME`

# change gpu to xpu in execution script
sed -i "s/\"gpu\"/\"xpu\"/g" test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo -e "\033[1;32m Started to run command: ${cmd}!  \033[0m"
eval $cmd
