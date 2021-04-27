
export CUDA_VISIBLE_DEVICES=0,5,6

# yml=fcn_hrnetw18_small_v1_humanseg_portrait
# save_dir=saved_model/${yml}
# yml=deeplabv3p_resnet50_os8_humanseg_512x512_100k
# save_dir=saved_model/${yml}
yml=shufflenetv2_humanseg_portrait
save_dir=saved_model/${yml}


# python train.py --config configs/${yml}.yml --do_eval --save_interval 30

mkdir -p ${save_dir}




# nohup python -u -m paddle.distributed.launch train.py --config configs/${yml}.yml --save_dir $save_dir \
# --save_interval 200 --num_workers 8 --do_eval \
# --use_vdl \
# 2>&1 | tee  -a ${save_dir}/log \

# python predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path data/matting_human_half/small_val.txt --save_dir ${save_dir}/small_val_predict

python predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
--image_path data/conference_human --save_dir ${save_dir}/conference_predict


# python -m paddle.distributed.launch predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path /ssd2/chulutao/humanseg/val.txt --save_dir ${save_dir}/all_val_predict
