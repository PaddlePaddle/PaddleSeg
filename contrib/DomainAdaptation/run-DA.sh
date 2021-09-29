export CUDA_VISIBLE_DEVICES=0,1,2,3

yml=deeplabv2_resnet101_os8_cityscapes_769x769_80k
save_dir=saved_model/${yml}
mkdir -p ${save_dir}

python -u -m paddle.distributed.launch --log_dir $save_dir train.py \
--config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 100 \
 --num_workers 3 --do_eval \
--keep_checkpoint_max 10 \
2>&1 | tee  ${save_dir}/log \

# python3 -m paddle.distributed.launch val.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml \
# --model_path saved_model/pointrend_resnet101_os8_cityscapes_1024×512_80k/best_model/model.pdparams --num_workers 4

#python3 -m paddle.distributed.launch val.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml \
#--model_path output/pointrend.pdparams --num_workers 4
