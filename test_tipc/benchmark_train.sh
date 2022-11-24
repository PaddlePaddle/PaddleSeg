#!/bin/bash
source test_tipc/common_func.sh

# set env
python=python
export str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
export frame_version=${str_tmp%%.post*}
export frame_commit=$(echo `${python} -c "import paddle;print(paddle.version.commit)"`)

# run benchmark sh
# Usage:
# bash run_benchmark_train.sh config.txt params
# or
# bash run_benchmark_train.sh config.txt

function func_parser_params(){
    strs=$1
    IFS="="
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_sed_params(){
    filename=$1
    line=$2
    param_value=$3
    params=`sed -n "${line}p" $filename`
    IFS=":"
    array=(${params})
    key=${array[0]}
    value=${array[1]}
    # [Bobholamovic] This if block results in --batch_size=benchmark in the second 
    # test_train_inference_python.sh call, so I comment it out.
    # if [[ $value =~ 'benchmark_train' ]];then
    #     IFS='='
    #     _val=(${value})
    #     param_value="${_val[0]}=${param_value}"
    # fi
    new_params="${key}:${param_value}"
    IFS=";"
    cmd="sed -i '${line}s/.*/${new_params}/' '${filename}'"
    eval $cmd
}

function set_gpu_id(){
    string=$1
    _str=${string:1:6}
    IFS="C"
    arr=(${_str})
    M=${arr[0]}
    P=${arr[1]}
    gn=`expr $P - 1`
    gpu_num=`expr $gn / $M`
    seq=`seq -s "," 0 $gpu_num`
    echo $seq
}

function get_repo_name(){
    IFS=";"
    cur_dir=$(pwd)
    IFS="/"
    arr=(${cur_dir})
    echo ${arr[-1]}
}

FILENAME=$1
# copy FILENAME as new
new_filename="./test_tipc/benchmark_train.txt"
cmd=`yes|cp $FILENAME $new_filename`
FILENAME=$new_filename
# MODE must be one of ['benchmark_train']
MODE=$2
PARAMS=$3
# bash test_tipc/benchmark_train.sh test_tipc/configs/det_mv3_db_v2_0/train_benchmark.txt  benchmark_train dynamic_bs8_null_DP_N1C1
IFS=$'\n'
# parser params from train_benchmark.txt
dataline=`cat $FILENAME`
# parser params
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")

# 获取benchmark_params所在的行数
line_num=`grep -n -w "train_benchmark_params" $FILENAME  | cut -d ":" -f 1`
# for train log parser
batch_size=$(func_parser_value "${lines[line_num]}")
line_num=`expr $line_num + 1`
fp_items=$(func_parser_value "${lines[line_num]}")
line_num=`expr $line_num + 1`
epoch=$(func_parser_value "${lines[line_num]}")

line_num=`expr $line_num + 1`
profile_option_key=$(func_parser_key "${lines[line_num]}")
profile_option_params=$(func_parser_value "${lines[line_num]}")
profile_option="${profile_option_key}:${profile_option_params}"

line_num=`expr $line_num + 1`
flags_value=$(func_parser_value "${lines[line_num]}")
# set flags
IFS=";"
flags_list=(${flags_value})
for _flag in ${flags_list[*]}; do
    cmd="export ${_flag}"
    eval $cmd
done

# set log_name
repo_name=$(get_repo_name )
SAVE_LOG=${BENCHMARK_LOG_DIR:-$(pwd)}   # */benchmark_log
mkdir -p "${SAVE_LOG}/benchmark_log/"
status_log="${SAVE_LOG}/benchmark_log/results.log"

# The number of lines in which train params can be replaced.
line_python=3
line_gpuid=4
line_precision=6
line_epoch=7
line_batchsize=9
line_profile=13
line_eval_py=24
line_export_py=30

func_sed_params "$FILENAME" "${line_eval_py}" "null"
func_sed_params "$FILENAME" "${line_export_py}" "null"
func_sed_params "$FILENAME" "${line_python}"  "$python"

# Parse extra args
parse_extra_args "${lines[@]}"
for params in ${extra_args[*]}; do
    IFS=":"
    arr=(${params})
    key=${arr[0]}
    value=${arr[1]}
    if [ "${key}" = 'skip_iters' ]; then
        skip_iters="${value}"
    elif [ "${key}" = 'task' ]; then
        task="${value}"
    fi
done

# if params
if  [ ! -n "$PARAMS" ] ;then
    # PARAMS input is not a word.
    IFS="|"
    batch_size_list=(${batch_size})
    fp_items_list=(${fp_items})
    device_num_list=(N1C1 N1C8)
    run_mode="DP"
elif [[ ${PARAMS} = "dynamicTostatic" ]] ;then
    IFS="|"
    model_type=$PARAMS
    batch_size_list=(${batch_size})
    fp_items_list=(${fp_items})
    device_num_list=(N1C1 N1C8)
    run_mode="DP"
else
    # parser params from input: modeltype_bs{fp_item}_{device_num}
    IFS="_"
    params_list=(${PARAMS})
    model_type=${params_list[0]}
    batch_size=${params_list[1]}
    batch_size=`echo  ${batch_size} | tr -cd "[0-9]" `
    precision=${params_list[2]}
    # run_process_type=${params_list[3]}
    run_mode=${params_list[3]}
    device_num=${params_list[4]}
    IFS=";"

    if [ ${precision} = "null" ];then
        precision="fp32"
    fi

    fp_items_list=($precision)
    batch_size_list=($batch_size)
    device_num_list=($device_num)
fi

# for log name
to_static=""
# parse "to_static" options and modify trainer into "to_static_trainer"
if [[ ${model_type} = "dynamicTostatic" ]];then
    to_static="d2sT_"
    sed -i 's/trainer:norm_train/trainer:to_static_train/g' $FILENAME
fi

IFS="|"
for batch_size in ${batch_size_list[*]}; do
    for precision in ${fp_items_list[*]}; do
        for device_num in ${device_num_list[*]}; do
            # sed batchsize and precision
            func_sed_params "$FILENAME" "${line_precision}" "$precision"
            func_sed_params "$FILENAME" "${line_batchsize}" "$MODE=$batch_size"
            func_sed_params "$FILENAME" "${line_epoch}" "$MODE=$epoch"
            gpu_id=$(set_gpu_id $device_num)

            if [ ${#gpu_id} -le 1 ];then
                log_path="$SAVE_LOG/profiling_log"
                mkdir -p $log_path
                log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_mode}_${device_num}_${to_static}profiling"
                func_sed_params "$FILENAME" "${line_gpuid}" "0"  # sed used gpu_id
                # set profile_option params
                tmp=`sed -i "${line_profile}s/.*/${profile_option}/" "${FILENAME}"`

                # run test_train_inference_python.sh
                cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} benchmark_train > ${log_path}/${log_name} 2>&1 "
                echo $cmd
                eval $cmd
                eval "cat ${log_path}/${log_name}"

                # without profile
                log_path="$SAVE_LOG/train_log"
                speed_log_path="$SAVE_LOG/index"
                mkdir -p $log_path
                mkdir -p $speed_log_path
                log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_mode}_${device_num}_${to_static}log"
                speed_log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_mode}_${device_num}_${to_static}speed"
                func_sed_params "$FILENAME" "${line_profile}" "null"  # sed profile_id as null
                cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} benchmark_train > ${log_path}/${log_name} 2>&1 "
                echo $cmd
                job_bt=`date '+%Y%m%d%H%M%S'`
                eval $cmd
                job_et=`date '+%Y%m%d%H%M%S'`
                export model_run_time=$((${job_et}-${job_bt}))
                eval "cat ${log_path}/${log_name}"

                # [Bobholamovic] For matting tasks, modify the training log to fit the input of analysis.py.
                if [ "${task}" = 'mat' ]; then
                    sed -i 's/=/: /g' ${log_path}/${log_name}
                fi

                if [ -n "${skip_iters}" ]; then
                    filtered_log_name=${log_name}_filtered
                    cmd="${python} test_tipc/filter_log.py \
                            --in_log_path '${log_path}/${log_name}' \
                            --out_log_path '${log_path}/${filtered_log_name}' \
                            --skip_iters ${skip_iters}"
                    echo $cmd
                    eval $cmd
                    log_name=${filtered_log_name}
                fi

                # parser log
                _model_name="${model_name}_bs${batch_size}_${precision}_${run_mode}"
                cmd="${python} ${BENCHMARK_ROOT}/scripts/analysis.py --filename ${log_path}/${log_name} \
                        --speed_log_file '${speed_log_path}/${speed_log_name}' \
                        --model_name ${_model_name} \
                        --base_batch_size ${batch_size} \
                        --run_mode ${run_mode} \
                        --fp_item ${precision} \
                        --keyword ips: \
                        --skip_steps 2 \
                        --device_num ${device_num} \
                        --speed_unit samples/s \
                        --convergence_key loss: "
                echo $cmd
                eval $cmd
                last_status=${PIPESTATUS[0]}
                status_check $last_status "${cmd}" "${status_log}"
            else
                IFS=";"
                unset_env=`unset CUDA_VISIBLE_DEVICES`
                log_path="$SAVE_LOG/train_log"
                speed_log_path="$SAVE_LOG/index"
                mkdir -p $log_path
                mkdir -p $speed_log_path
                log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_mode}_${device_num}_${to_static}log"
                speed_log_name="${repo_name}_${model_name}_bs${batch_size}_${precision}_${run_mode}_${device_num}_${to_static}speed"
                func_sed_params "$FILENAME" "${line_gpuid}" "$gpu_id"  # sed used gpu_id
                func_sed_params "$FILENAME" "${line_profile}" "null"  # sed --profile_option as null
                cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} benchmark_train > ${log_path}/${log_name} 2>&1 "
                echo $cmd
                job_bt=`date '+%Y%m%d%H%M%S'`
                eval $cmd
                job_et=`date '+%Y%m%d%H%M%S'`
                export model_run_time=$((${job_et}-${job_bt}))
                eval "cat ${log_path}/${log_name}"

                # [Bobholamovic] For matting tasks, modify the training log to fit the input of analysis.py.
                if [ "${model_name}" = 'ppmatting' ]; then
                    sed -i 's/=/: /g' ${log_path}/${log_name}
                fi

                # parser log
                _model_name="${model_name}_bs${batch_size}_${precision}_${run_mode}"
                
                cmd="${python} ${BENCHMARK_ROOT}/scripts/analysis.py --filename ${log_path}/${log_name} \
                        --speed_log_file '${speed_log_path}/${speed_log_name}' \
                        --model_name ${_model_name} \
                        --base_batch_size ${batch_size} \
                        --run_mode ${run_mode} \
                        --fp_item ${precision} \
                        --keyword ips: \
                        --skip_steps 2 \
                        --device_num ${device_num} \
                        --speed_unit images/s \
                        --convergence_key loss: "
                echo $cmd
                eval $cmd
                last_status=${PIPESTATUS[0]}
                status_check $last_status "${cmd}" "${status_log}" "${model_name}"
            fi
        done
    done
done
