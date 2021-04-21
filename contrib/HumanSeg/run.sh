
export CUDA_VISIBLE_DEVICES=5,6,7

# yml=fcn_hrnetw18_small_v1_humanseg_192x192
# save_dir=saved_model/${yml}_bs64_lr0.1_iter2w_horizontal_distort
yml=deeplabv3p_resnet50_os8_humanseg_512x512_100k
save_dir=saved_model/${yml}


# python train.py --config configs/${yml}.yml --do_eval --save_interval 30

mkdir -p ${save_dir}


python -m paddle.distributed.launch predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
--image_path /ssd2/chulutao/humanseg/val.txt --save_dir ${save_dir}/all_val_predict


# nohup python -u -m paddle.distributed.launch train.py --config configs/${yml}.yml --save_dir $save_dir \
# --save_interval 200 --num_workers 8 --do_eval \
# --use_vdl \
# 2>&1 | tee  -a ${save_dir}/log \

# python predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path /ssd2/chulutao/humanseg/small_val.txt --save_dir ${save_dir}/val_predict
