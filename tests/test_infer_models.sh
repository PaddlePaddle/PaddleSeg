# This file tests all pretrained inference model on GPU, outputs the accuracy and speed.
# Use tests/analyze_infer_models_log.py to analyze the output logs.

# Usage:
#   1. Install PaddlePaddle that supports TenorRT
#   2. `export CUDA_VISIBLE_DEVICES=id`
#   3. `cd ./PaddleSeg`
#   4. `nohup bash ./tests/test_infer_models.sh /path/to/cityscapes 2>&1 >logs.txt &`
#   5. `python tests/analyze_infer_models_log.py --log_path ./logs.txt --save_path ./info.txt`

if [ $# != 1 ] ; then
    echo "USAGE: $0 cityscapes_dataset_path"
    exit 1;
fi

dataset_path=$1 # Dataset path for infer_benchmark.py
echo "dataset_path: ${dataset_path}"
dataset_type="Cityscapes"   # Dataset type for infer_benchmark.py

pretrained_root_path="./pretrained_model"   # the root path for saving pretrained model
download_root_path="https://bj.bcebos.com/paddleseg/dygraph/cityscapes"
config_root_path="./configs"
save_root_path="./output_tmp"  # the root path for saving inference model
save_basename="fp32_infer" # the basename for saving inference model
enable_auto_tune=True   # Use auto tune for GPU TRT inference
resize_width=0  # 0 means not use resize
resize_height=0  # 0 means not use resize

mkdir -p ${pretrained_root_path}
mkdir -p ${save_root_path}

# collect model configs_path that have pretrained weights of cityscapes dataset
echo -e "Collect model configs"
configs_path=()
all_files=`find  $config_root_path -name "*.yml"`
all_files=$(echo ${all_files[*]} | tr ' ' '\n' | sort -n)
for config_path in ${all_files[@]}
do
    model_name=$(basename ${config_path} .yml)
    download_path="${download_root_path}/${model_name}/model.pdparams"
    urlstatus=$(curl -s -m 5 -IL $download_path | grep 200)
    if [ "$urlstatus" != "" ] && [[ $config_path =~ "cityscapes" ]];then
        configs_path[${#configs_path[@]}]=${config_path}
    fi
done
echo -e "configs_path: ${configs_path[*]} \n"

# test all models
for config_path in ${configs_path[@]}
do
    model_name=$(basename ${config_path} .yml)

    echo "config_path: ${config_path}"
    echo "model_name: ${model_name}"

    download_path=${download_root_path}/${model_name}/model.pdparams
    pretrained_path=${pretrained_root_path}/${model_name}.pdparams
    if [ ! -f ${pretrained_path} ];then
        echo -e "\n Download pretrained weights"
        wget ${download_path} -O ${pretrained_path}
    fi

    echo -e "\n Analyze model"
    python tools/analyze_model.py \
        --config ${config_path}

    echo -e "\n Export inference model"
    export_path=${save_root_path}/${model_name}/${save_basename}
    if [ -d ${export_path} ]; then
        rm -rf ${export_path}
    fi
    python ./export.py \
        --config ${config_path}\
        --model_path ${pretrained_path} \
        --save_dir ${export_path}

    echo -e "\n Test ${model_name} GPU Naive fp32"
    python deploy/python/infer_benchmark.py \
        --dataset_type ${dataset_type} \
        --dataset_path ${dataset_path} \
        --device cpu \
        --use_trt False \
        --precision fp32 \
        --resize_width ${resize_width} \
        --resize_height ${resize_height} \
        --config ${export_path}/deploy.yaml

    echo -e "\n Test ${model_name} GPU TRT fp32"
    python deploy/python/infer_benchmark.py \
        --dataset_type ${dataset_type} \
        --dataset_path ${dataset_path} \
        --device gpu \
        --use_trt True \
        --precision fp32 \
        --enable_auto_tune ${enable_auto_tune} \
        --resize_width ${resize_width} \
        --resize_height ${resize_height} \
        --config ${export_path}/deploy.yaml

    echo -e "\n Test ${model_name} GPU TRT fp16"
    python deploy/python/infer_benchmark.py \
        --dataset_type ${dataset_type} \
        --dataset_path ${dataset_path} \
        --device gpu \
        --use_trt True \
        --precision fp16 \
        --enable_auto_tune ${enable_auto_tune} \
        --resize_width ${resize_width} \
        --resize_height ${resize_height} \
        --config ${export_path}/deploy.yaml

    echo -e "\n Test ${model_name} CPU Naive"
    python deploy/python/infer_benchmark.py \
        --dataset_type ${dataset_type} \
        --dataset_path ${dataset_path} \
        --device cpu \
        --enable_mkldnn False \
        --resize_width ${resize_width} \
        --resize_height ${resize_height} \
        --config ${export_path}/deploy.yaml

    echo -e "\n Test ${model_name} CPU MKLDNN"
    python deploy/python/infer_benchmark.py \
        --dataset_type ${dataset_type} \
        --dataset_path ${dataset_path} \
        --device cpu \
        --enable_mkldnn True \
        --resize_width ${resize_width} \
        --resize_height ${resize_height} \
        --config ${export_path}/deploy.yaml

    echo -e "\n\n"
done
