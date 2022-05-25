echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "model: ${model}"
echo "tag: ${tag}"

save_dir="output/${model}/${tag}"
mkdir -p ${save_dir}

echo "config: configs/${model}.yml"
echo "save_dir: ${save_dir}"
echo "train log: ${save_dir}/log_train.txt"

nohup python -m paddle.distributed.launch --log_dir ${save_dir}/log_dir train.py \
    --config configs/${model}.yml \
    --save_dir ${save_dir} \
    --save_interval 500 \
    --num_workers 3 \
    --do_eval \
    --use_vdl \
    --log_iters 10 \
    2>&1 > ${save_dir}/log_train.txt &
