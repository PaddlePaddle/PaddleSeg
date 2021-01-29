ulimit -u 20000
ulimit -a
export CUDA_VISIBLE_DEVICES=7

yml=mscale_ocr_cityscapes_autolabel_mapillary_dice_boot_mink10w_lr0.005
save_dir=saved_model/$yml
mkdir -p ${save_dir}


# nohup python -u -m paddle.distributed.launch train.py --config configs/ocrnet/${yml}.yml --use_vdl --save_dir $save_dir --save_interval 2000 --num_workers 5 --do_eval \
# 2>&1 | tee  ${save_dir}/log \
# && python -u -m paddle.distributed.launch ms_val.py --config configs/ocrnet/${yml}.yml \
# --model_path ${save_dir}/best_model/model.pdparams \
# --num_workers 5 \
# 2>&1 | tee  ${save_dir}/ms_flip_eval_log


# echo "++++++++++++++++++++++++ predict test"
# root_dir=${save_dir}/testset_flip_predict_results
# img_dir=/ssd1/home/chulutao/dataset/cityscapes-for-nvidia/leftImg8bit/test

# python  -u -m paddle.distributed.launch predict.py --config configs/ocrnet/${yml}.yml \
# --model_path ${save_dir}/best_model/model.pdparams \
# --image_path ${img_dir} \
# --save_dir ${root_dir} \
# --aug_pred \
# --flip_horizontal \
# && \
# python convert_cityscapes_trainid2labelid.py --root_dir ${root_dir} \
# && \
# cd ${root_dir} && zip -r convert_to_labelid.zip  convert_to_labelid/ && cd ../../..


# python train.py --config configs/${yml}.yml --save_dir $save_dir --save_interval 20 --num_workers 5 --do_eval

# python val.py --config configs/${yml}.yml

python predict.py --config configs/${yml}.yml

# python ms_val.py --config configs/ocrnet/${yml}.yml \
# --model_path ${save_dir}/best_model/model.pdparams \
# --num_workers 5 \
