echo "model: ${model}"
echo "tag: ${tag}"

config="configs/${model}.yml"
save_dir="output/${model}/${tag}"
model_path="${save_dir}/best_model/model.pdparams"

echo "config: ${config}"
echo "save_dir: ${save_dir}"
echo "model_path: ${model_path}"

python ./scripts/mixed_data_predict.py \
    --config ${config} \
    --model_path=${model_path} \
    --file_list=small_test.txt \
    --save_dir=${save_dir}
