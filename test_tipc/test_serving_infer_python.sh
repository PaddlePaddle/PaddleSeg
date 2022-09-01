#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
dataline=$(awk 'NR==1, NR==18{print}'  $FILENAME)
MODE=$2

# parser params
IFS=$'\n'
lines=(${dataline})

# parser serving
model_name=$(func_parser_value "${lines[1]}")
python_list=$(func_parser_value "${lines[2]}")
trans_model_py=$(func_parser_value "${lines[3]}")
infer_model_dir_key=$(func_parser_key "${lines[4]}")
infer_model_dir_value=$(func_parser_value "${lines[4]}")
model_filename_key=$(func_parser_key "${lines[5]}")
model_filename_value=$(func_parser_value "${lines[5]}")
params_filename_key=$(func_parser_key "${lines[6]}")
params_filename_value=$(func_parser_value "${lines[6]}")
serving_server_key=$(func_parser_key "${lines[7]}")
serving_server_value=$(func_parser_value "${lines[7]}")
serving_client_key=$(func_parser_key "${lines[8]}")
serving_client_value=$(func_parser_value "${lines[8]}")
serving_dir_value=$(func_parser_value "${lines[9]}")
web_service_py=$(func_parser_value "${lines[10]}")
web_use_gpu_key=$(func_parser_key "${lines[11]}")
web_use_gpu_list=$(func_parser_value "${lines[11]}")
output_name_key=$(func_parser_key "${lines[12]}")
output_name_value=$(func_parser_value "${lines[12]}")
pipeline_py=$(func_parser_value "${lines[13]}")
image_dir_key=$(func_parser_key "${lines[14]}")
image_dir_value=$(func_parser_value "${lines[14]}")
input_name_key=$(func_parser_key "${lines[15]}")
input_name_value=$(func_parser_value "${lines[15]}")


LOG_PATH="./test_tipc/output/${model_name}/${MODE}"  ##
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

function func_serving(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    # pdserving
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_input_name=$(func_set_params "${input_name_key}" "${input_name_value}")
    set_output_name=$(func_set_params "${output_name_key}" "${output_name_value}")
    set_serving_server=$(func_set_params "${serving_server_key}" "${serving_server_value}")
    set_serving_client=$(func_set_params "${serving_client_key}" "${serving_client_value}")
    python_list=(${python_list})
    python=${python_list[0]}
    trans_model_log_path="${LOG_PATH}/python_trans_model.log"
    trans_model_cmd="${python} ${trans_model_py} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_serving_server} ${set_serving_client} >${trans_model_log_path} 2>&1"
    eval ${trans_model_cmd}
    last_status=${PIPESTATUS[0]}
    cd ${serving_dir_value}
    status_check $last_status "${trans_model_cmd}" "${status_log}" "${model_name}" "${trans_model_log_path}"
    echo $PWD

    for use_gpu in ${web_use_gpu_list[*]}; do
        if [ ${use_gpu} = "null" ]; then
            web_service_log_path="${LOG_PATH}/python_server_cpu.log"
            web_service_cmd="${python} ${web_service_py} ${set_input_name} ${set_output_name} ${web_use_gpu_key}="" >${web_service_log_path} 2>&1 &"
            eval $web_service_cmd
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}" "${web_service_log_path}"
            sleep 5s
            client_log_path="${LOG_PATH}/python_client_cpu_batchsize_1.log"
            set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            pipeline_cmd="${python} ${pipeline_py} ${set_image_dir}  ${set_input_name} > ${client_log_path} 2>&1 "
            eval $pipeline_cmd
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}" "${client_log_path}"
            eval "cat ${client_log_path}"
            ps ux | grep -E 'web_service' | awk '{print $2}' | xargs kill -s 9
        else
            web_service_log_path="${LOG_PATH}/python_server_gpu.log"
            web_service_cmd="${python} ${web_service_py} ${set_input_name} ${set_output_name} ${web_use_gpu_key}=${use_gpu} >${web_service_log_path} 2>&1 &"
            eval $web_service_cmd
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}" "${web_service_log_path}"
            sleep 5s
            client_log_path="${LOG_PATH}/python_client_gpu_batchsize_1.log"
            set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
            pipeline_cmd="${python} ${pipeline_py} ${set_image_dir} ${set_input_name} > ${client_log_path} 2>&1 "
            eval $pipeline_cmd
            last_status=${PIPESTATUS[0]}
            status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}" "${client_log_path}"
            eval "cat ${client_log_path}"
            ps ux | grep -E 'web_service' | awk '{print $2}' | xargs kill -s 9
        fi
    done
}


# set cuda device
GPUID=$3
if [ ${#GPUID} -le 0 ];then
    env=" "
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
set CUDA_VISIBLE_DEVICES
eval $env


echo "################### run test ###################"

export Count=0
IFS="|"
func_serving "${web_service_cmd}"
