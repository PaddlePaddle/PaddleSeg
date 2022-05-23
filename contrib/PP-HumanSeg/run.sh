ulimit -u 40960
ulimit -u

export CUDA_VISIBLE_DEVICES=2,3

# experiment=pphumanseg_lite_imagenet+humanseg_pretrain_mixed_train_SCL0.1
# save_dir=saved_model/${experiment}
# cfg=configs/${experiment}.yml
# save_interval=`expr 10000 / 25`


################## test mode
# python mixed_data_train.py --config ${cfg} --do_eval \
# --log_iters 1 --save_interval 10 --num_workers=4 

# python mixed_data_val.py --config ${cfg} \
# --model_path=${save_dir}/best_model/model.pdparams \
# --file_list=test.txt \
# --num_workers=4


################# mixed train
# mkdir -p ${save_dir}
# cp run.sh ${save_dir}


# nohup python -u -m paddle.distributed.launch --log_dir ${save_dir}/   \
# mixed_data_train.py  --config ${cfg} \
# --save_dir $save_dir \
# --num_workers 3 --do_eval \
# --use_vdl \
# 2>&1 | tee  ${save_dir}/log


# python -u -m paddle.distributed.launch mixed_data_val.py \
# --config ${cfg} \
# --model_path=${save_dir}/best_model/model.pdparams \
# --file_list=test.txt \
# --num_workers 4 \
# 2>&1 | tee  ${save_dir}/eval_testset_log

# python -u -m paddle.distributed.launch mixed_data_predict.py \
# --config ${cfg} \
# --model_path=${save_dir}/best_model/model.pdparams \
# --file_list=small_test.txt \
# --save_dir=${save_dir}


experiment=pphumanseg_lite_imagenet+humanseg_pretrain_mixed_train_SCL0.5
save_dir=saved_model_jc/${experiment}
cfg=configs/${experiment}.yml
save_interval=`expr 10000 / 25`
################# mixed train
mkdir -p ${save_dir}


nohup python -u -m paddle.distributed.launch --log_dir ${save_dir}/logs \
mixed_data_train.py  --config ${cfg} \
--save_dir $save_dir \
--num_workers 3 \
--do_eval \
--use_vdl \
2>&1 | tee  ${save_dir}/log.txt


python -u -m paddle.distributed.launch mixed_data_val.py \
--config ${cfg} \
--model_path=${save_dir}/best_model/model.pdparams \
--file_list=test.txt \
--num_workers 4 \
2>&1 | tee  ${save_dir}/eval_testset_log.txt

python -u -m paddle.distributed.launch mixed_data_predict.py \
--config ${cfg} \
--model_path=${save_dir}/best_model/model.pdparams \
--file_list=small_test.txt \
--save_dir=${save_dir}
