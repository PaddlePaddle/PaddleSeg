#!/bin/bash
source test_tipc/common_func.sh


FILENAME=$1
INFERIMG=$2

dataline=$(cat ${FILENAME})
lines=(${dataline})

# parser params
dataline=$(awk 'NR==1, NR==14{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser load config
model_name=$(func_parser_value_cpp "${lines[1]}")
use_gpu_key=$(func_parser_key_cpp "${lines[2]}")
use_gpu_value=$(func_parser_value_cpp "${lines[2]}")
use_mkldnn_key=$(func_parser_key_cpp "${lines[6]}")
use_mkldnn_value=$(func_parser_value_cpp "${lines[6]}")
use_tensorrt_key=$(func_parser_key_cpp "${lines[7]}")
use_tensorrt_value=$(func_parser_value_cpp "${lines[7]}")
use_fp16_key=$(func_parser_key_cpp "${lines[8]}")
use_fp16_value=$(func_parser_value_cpp "${lines[8]}")

LOG_PATH="./test_tipc/output/infer_cpp"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_infer_cpp.log"

function func_infer_cpp(){
    # inference cpp
    if test $use_gpu_value -gt 0; then
        if test $use_tensorrt_value -gt 0; then
            if test $use_fp16_value -gt 0; then
                _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}_${use_tensorrt_key}_${use_fp16_key}.log"
            else
                _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}_${use_tensorrt_key}.log"
            fi
        else
            _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}.log"
        fi
    else
        if test $use_mkldnn_value -gt 0; then
            _save_log_path="${LOG_PATH}/infer_cpp_use_cpu_${use_mkldnn_key}.log"
        else
            _save_log_path="${LOG_PATH}/infer_cpp_use_cpu.log"
        fi    
    fi
    # run infer cpp
    inference_cpp_cmd="./test_tipc/cpp/build/seg_system"
    infer_cpp_full_cmd="${inference_cpp_cmd} ${FILENAME} ${INFERIMG} > ${_save_log_path} 2>&1 "   
    eval $infer_cpp_full_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${infer_cpp_full_cmd}" "${status_log}" "${model_name}"
}

echo "################### run test cpp inference ###################"

func_infer_cpp 