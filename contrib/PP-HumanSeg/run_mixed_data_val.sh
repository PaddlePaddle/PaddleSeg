echo "model: ${model}"
echo "tag: ${tag}"

config="configs/${model}.yml"
save_dir="output/${model}/${tag}"
model_path="${save_dir}/best_model/model.pdparams"
output_log="${save_dir}/log_test.txt"

echo "config: ${config}"
echo "save_dir: ${save_dir}"
echo "model_path: ${model_path}"
echo "output log: ${output_log}"

python mixed_data_val.py \
    --config=${config} \
    --model_path=${model_path} \
    --file_list=test.txt \
    --num_workers 4 \
    2>&1 | tee  ${output_log}
