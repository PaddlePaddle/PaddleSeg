export CUDA_VISIBLE_DEVICES=7

yml=deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_featpullin
save_dir=saved_model_develop/${yml}_1103
mkdir -p ${save_dir}

python train.py \
--config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 100 \
--num_workers 4 --do_eval \
--keep_checkpoint_max 10  --seed 42 \
2>&1 | tee  ${save_dir}/log \
