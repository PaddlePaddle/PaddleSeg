#!/bin/bash

function func_parser_key(){
    local strs=$1
    local IFS=":"
    local array=(${strs})
    local tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    local strs=$1
    local IFS=":"
    local array=(${strs})
    local tmp=${array[1]}
    echo ${tmp}
}

function func_parser_key_cpp(){
    local strs=$1
    local IFS=" "
    local array=(${strs})
    local tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value_cpp(){
    local strs=$1
    local IFS=" "
    local array=(${strs})
    local tmp=${array[1]}
    echo ${tmp}
}

function func_set_params(){
    local key=$1
    local value=$2
    if [ ${key}x = "null"x ]; then
        echo " "
    elif [[ ${value} = "null" ]] || [[ ${value} = " " ]] || [ ${#value} -le 0 ]; then
        echo " "
    else
        echo "${key}=${value}"
    fi
}

function func_parser_params(){
    local strs=$1
    local IFS=":"
    local array=(${strs})
    local key=${array[0]}
    local tmp=${array[1]}
    local IFS="|"
    local res=""
    for _params in ${tmp[*]}; do
        local IFS="="
        local array=(${_params})
        local mode=${array[0]}
        local value=${array[1]}
        if [[ ${mode} = ${MODE} ]]; then
            local IFS="|"
            #echo (funcsetparams"{mode}" "${value}")
            echo $value
            break
        fi
        local IFS="|"
    done
    echo ${res}
}

function status_check(){
    local last_status=$1   # the exit code
    local run_command=$2
    local run_log=$3
    local model_name=$4
    local log_path=$5
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${model_name} - ${run_command} - ${log_path}  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${model_name} - ${run_command} - ${log_path}  \033[0m" | tee -a ${run_log}
    fi
}

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" = "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

function parse_extra_args() {
    local lines=("$@")
    local last_idx=$((${#lines[@]}-1))
    local IFS=';'
    extra_args=(${lines[last_idx]})
}

function add_suffix() {
    local ori_path="$1"
    local suffix=$2
    local ext="${ori_path##*.}"
    echo "${ori_path%.*}${suffix}.${ext}"
}

function run_command() {
    local cmd="$1"
    local log_path="$2"
    if [ -n "${log_path}" ]; then
        eval ${cmd} 2>&1 | tee "${log_path}"
        test ${PIPESTATUS[0]} -eq 0
    else
        eval ${cmd}
    fi
}
