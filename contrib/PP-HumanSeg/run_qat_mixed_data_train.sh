echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

model=1_mixed_dataset_mobileseg_mobilenetv3_x4
tag=test_qat_0
model_path=output/1_mixed_dataset_mobileseg_mobilenetv3_x4/test_1/best_model/model.pdparams
lr=0.001
iters=100000

save_dir="output/${model}/${tag}"
mkdir -p ${save_dir}
cd ${save_dir}
rm -rf log_dir
rm -rf log_train.txt
rm -rf *.log
cd -

echo "config: configs/${model}.yml"
echo "save_dir: ${save_dir}"
echo "train log: ${save_dir}/log_dir/"

nohup python -m paddle.distributed.launch --log_dir ${save_dir}/log_dir ./scripts/qat_mixed_data_train.py \
    --config configs/${model}.yml \
    --model_path ${model_path} \
    --iters ${iters} \
    --learning_rate ${lr} \
    --save_dir ${save_dir} \
    --num_workers 3 \
    --do_eval \
    --use_vdl \
    --log_iters 10 \
    2>&1 > ${save_dir}/log_train.txt &

