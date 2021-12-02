export CUDA_VISIBLE_DEVICES=2

yml=deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_gta5src
save_dir=saved_model_develop/${yml}_test
mkdir -p ${save_dir}

python train.py \
--config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 30 \
--num_workers 4 --do_eval \
--keep_checkpoint_max 10  --seed 1234 \
2>&1 | tee  ${save_dir}/log \
