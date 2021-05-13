
ulimit -u 40960
ulimit -u

export CUDA_VISIBLE_DEVICES=6,7

# yml=fcn_hrnetw18_small_v1_humanseg_portrait
# save_dir=saved_model/${yml}
# yml=deeplabv3p_resnet50_os8_humanseg_512x512_100k
# save_dir=saved_model/${yml}
# yml=shufflenetv2_humanseg_398x224_aspect_padding
yml=shufflenetv2_humanseg_matting_portrait2600_398x224_iter7k_2gpu_lr0.1_pretrain
save_dir=saved_model/${yml}


# python train.py --do_eval --save_interval 30 --iters 20 --batch_size 10
# python train.py --config configs/${yml}.yml --do_eval --save_interval 30 --iters 40 --batch_size 80



# mkdir -p ${save_dir}


nohup python -u -m paddle.distributed.launch train.py  --save_dir $save_dir \
--num_workers 4 --do_eval \
--use_vdl \
2>&1 | tee  -a ${save_dir}/log


# nohup python -u -m paddle.distributed.launch train.py --config configs/${yml}.yml --save_dir $save_dir \
# --save_interval 50 --num_workers 8 --do_eval \
# --use_vdl \
# 2>&1 | tee  -a ${save_dir}/log

# python predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path data/matting_human_half/small_val.txt --save_dir ${save_dir}/small_val_predict

# python predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path data/conference_human --save_dir ${save_dir}/conference_predict


# python -m paddle.distributed.launch predict.py --config configs/${yml}.yml --model_path ${save_dir}/best_model/model.pdparams \
# --image_path data/portrait2600/val.txt --save_dir ${save_dir}/val_predict
