#export CUDA_VISIBLE_DEVICES=0,1

experiment=pphumanseg_lite_pretrain2_mixed_train_SCL0.5
save_dir=output/${experiment}/test_0
cfg=configs/${experiment}.yml
save_interval=`expr 10000 / 25`
mkdir -p ${save_dir}

# train
nohup python -u -m paddle.distributed.launch --log_dir ${save_dir}/logs \
mixed_data_train.py  --config ${cfg} \
--save_dir $save_dir \
--num_workers 3 \
--do_eval \
--use_vdl \
2>&1 | tee  ${save_dir}/log.txt

# val
python -u -m paddle.distributed.launch mixed_data_val.py \
--config ${cfg} \
--model_path=${save_dir}/best_model/model.pdparams \
--file_list=test.txt \
--num_workers 4 \
2>&1 | tee  ${save_dir}/eval_testset_log.txt

# predict
python -u -m paddle.distributed.launch mixed_data_predict.py \
--config ${cfg} \
--model_path=${save_dir}/best_model/model.pdparams \
--file_list=small_test.txt \
--save_dir=${save_dir}
