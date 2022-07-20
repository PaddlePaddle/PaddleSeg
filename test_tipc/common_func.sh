#!/bin/bash

function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_parser_key_cpp(){
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value_cpp(){
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_set_params(){
    key=$1
    value=$2
    if [ ${key}x = "null"x ];then
        echo " "
    elif [[ ${value} = "null" ]] || [[ ${value} = " " ]] || [ ${#value} -le 0 ];then
        echo " "
    else
        echo "${key}=${value}"
    fi
}

function func_parser_params(){
    strs=$1
    IFS=":"
    array=(${strs})
    key=${array[0]}
    tmp=${array[1]}
    IFS="|"
    res=""
    for _params in ${tmp[*]}; do
        IFS="="
        array=(${_params})
        mode=${array[0]}
        value=${array[1]}
        if [[ ${mode} = ${MODE} ]]; then
            IFS="|"
            #echo (funcsetparams"{mode}" "${value}")
            echo $value
            break
        fi
        IFS="|"
    done
    echo ${res}
}

function status_check(){
    local last_status=$1   # the exit code
    local run_command=$2
    local run_log=$3
    local model_name=$4
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${model_name} - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${model_name} - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}
