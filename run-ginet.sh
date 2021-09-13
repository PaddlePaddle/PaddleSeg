export CUDA_VISIBLE_DEVICES=2,3,4,5


yml=ginet_resnet101_os8_pascal_context_80k
save_dir=saved_model/${yml}
mkdir -p ${save_dir}


python -u -m paddle.distributed.launch --log_dir $save_dir train.py \
--config configs/ginet/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 100 \
--num_workers 3 --do_eval \
--keep_checkpoint_max 10 \
2>&1 | tee  ${save_dir}/log \

