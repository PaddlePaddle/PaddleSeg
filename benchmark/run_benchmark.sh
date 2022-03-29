#!/usr/bin/env bash
set -xe

# Test training benchmark for a model.

# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${max_iter} ${model_name} ${num_workers}

function _set_params(){
    run_mode=${1:-"sp"}         # sp or mp
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}        # fp32 or fp16
    max_iter=${4:-"100"}
    model_item=${5:-"model_item"}   # fastscnn|segformer_b0| ocrnet_hrnetw48
    num_workers=${6:-"3"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    base_batch_size=${batch_size}
    mission_name="图像分割"
    direction_id="0"
    ips_unit="samples/sec"
    skip_steps=10                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    keyword_loss="loss:" #选填
    index="1"
    model_name=${model_item}_bs${batch_size}_${fp_item}  # 模型的不同bs、fp配置模型应不一样,避免入库会混乱

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--precision fp16"
    fi
    train_cmd="--config=benchmark/configs/${model_item}.yml \
               --batch_size=${batch_size} \
               --iters=${max_iter} \
               --num_workers=${num_workers} ${use_fp16_cmd}"

    case ${run_mode} in
    sp) train_cmd="python -u train.py ${train_cmd}" ;;
    mp)
        rm -rf ./mylog
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

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;该脚本在连调时可从benchmark repo中下载https://github.com/PaddlePaddle/benchmark/blob/master/scripts/run_model.sh;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
# _train       # 如果只想产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只想要产出训练log可以注掉本行,提交时需打开
