#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==32{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
use_gpu_key=$(func_parser_key "${lines[3]}")
use_gpu_value=$(func_parser_value "${lines[3]}")
model_path_key=$(func_parser_key "${lines[4]}")
model_path_value=$(func_parser_value "${lines[4]}")
output_dir_key=$(func_parser_key "${lines[5]}")
output_dir_value=$(func_parser_value "${lines[5]}")
data_dir_key=$(func_parser_key "${lines[6]}")
data_dir_value=$(func_parser_value "${lines[6]}")
batch_num_key=$(func_parser_key "${lines[7]}")
batch_num_value=$(func_parser_value "${lines[7]}")
batch_size_key=$(func_parser_key "${lines[8]}")
batch_size_value=$(func_parser_value "${lines[8]}")

# parser trainer
train_py=$(func_parser_value "${lines[11]}")

# parser inference 
inference_py=$(func_parser_value "${lines[14]}")
use_gpu_key=$(func_parser_key "${lines[15]}")
use_gpu_list=$(func_parser_value "${lines[15]}")
batch_size_key=$(func_parser_key "${lines[16]}")
batch_size_list=$(func_parser_value "${lines[16]}")
config_key=$(func_parser_key "${lines[17]}")
config_value=$(func_parser_value "${lines[17]}")
image_dir_key=$(func_parser_key "${lines[18]}")
infer_img_dir=$(func_parser_value "${lines[18]}")
benchmark_key=$(func_parser_key "${lines[19]}")
benchmark_value=$(func_parser_value "${lines[19]}")


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    # config_value=$3
    _log_path=$4
    _img_dir=$5
    # inference
    for use_gpu in ${use_gpu_list[*]}; do
        # cpu
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for batch_size in ${batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_cpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                set_model_dir=$(func_set_params "${config_key}" "${config_value}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                echo $command
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}" "${model_name}"
            done
        # gpu        
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for batch_size in ${batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_gpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                set_model_dir=$(func_set_params "${config_key}" "${config_value}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "               
                echo $command
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}"  "${model_name}" 
            done      
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

# log
LOG_PATH="./log/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

if [ ${MODE} = "whole_infer" ]; then
    IFS="|"
    # run export
    set_output_dir=$(func_set_params "${output_dir_key}" "${output_dir_value}")
    set_data_dir=$(func_set_params "${data_dir_key}" "${data_dir_value}")
    set_batch_size=$(func_set_params "${batch_size_key}" "${batch_size_value}")
    set_batch_num=$(func_set_params "${batch_num_key}" "${batch_num_value}")
    set_model_path=$(func_set_params "${model_path_key}" "${model_path_value}")
    set_use_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu_value}")
    
    export_cmd="${python} ${train_py} ${set_use_gpu} ${set_model_path} ${set_batch_num} ${set_batch_size} ${set_data_dir} ${set_output_dir}"
    echo $export_cmd
    eval $export_cmd
    status_export=$?
    status_check $status_export "${export_cmd}" "${status_log}" "${model_name}"
    
    save_infer_dir=${output_dir_value}
    #run inference
    func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_img_dir}"

fi