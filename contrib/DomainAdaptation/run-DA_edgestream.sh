export CUDA_VISIBLE_DEVICES=5

yml=deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_edgestream
save_dir=saved_model_develop/${yml}
mkdir -p ${save_dir}

python train.py \
--config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 300 \
 --num_workers 4 --do_eval \
--keep_checkpoint_max 10  --seed 0 \
2>&1 | tee  ${save_dir}/log \

# python -u -m paddle.distributed.launch --log_dir $save_dir train.py \
# --config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
# --save_interval 1000 --log_iters 100 \
#  --num_workers 3 --do_eval \
# --keep_checkpoint_max 10 \
# 2>&1 | tee  ${save_dir}/log \

# python3 val.py \
# --config configs/deeplabv2/${yml}.yml \
# --model_path models/torch_transfer_trained.pdparams --num_workers 4

# python3 -m paddle.distributed.launch val.py \
# --config configs/deeplabv2/${yml}.yml \
# --model_path models/torch_transfer_gta5source.pdparams --num_workers 4
