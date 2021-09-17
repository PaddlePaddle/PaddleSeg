#!/usr/bin/env bash
set -xe

# Test training benchmark for a model.

# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${max_iter} ${model_name} ${num_workers}

function _set_params(){
    run_mode=${1:-"sp"}         # sp or mp
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}        # fp32 or fp16
    max_iter=${4:-"100"}
    model_name=${5:-"model_name"}
    num_workers=${6:-"3"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--fp16"
    fi
    train_cmd="--config=benchmark/configs/${model_name}.yml \
               --batch_size=${batch_size} \
               --iters=${max_iter} \
               --num_workers=${num_workers} ${use_fp16_cmd}"

    case ${run_mode} in
    sp) train_cmd="python -u train.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  train.py ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params $@
_train
