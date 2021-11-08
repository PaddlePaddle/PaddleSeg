export CUDA_VISIBLE_DEVICES=6

yml=deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_test
save_dir=saved_model_develop/${yml}_edgeconvx4_1105
mkdir -p ${save_dir}

python train.py \
--config configs/deeplabv2/${yml}.yml --use_vdl --save_dir $save_dir  \
--save_interval 1000 --log_iters 300 \
--num_workers 4 --do_eval \
--resume_model  saved_model_develop/deeplabv2_resnet101_os8_gta5cityscapes_1280x640_160k_newds_test_edgeconvx4_1105/iter_122000/ \
--keep_checkpoint_max 10  --seed 42 \
2>&1 | tee  ${save_dir}/log \
