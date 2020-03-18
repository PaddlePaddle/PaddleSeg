#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,7
# export FLAGS_fraction_of_gpu_memory_to_use=0.0 # 统计模型真实显存消耗，按需分配
export FLAGS_fraction_of_gpu_memory_to_use=0.98 # 加大显存分配，默认0.92，最高为1.00

gpus=$(expr length $CUDA_VISIBLE_DEVICES)
gpus=$(expr $(expr $gpus + 1) / 2)



cfg="ce-voc-no-flip-large-stepscaling"
config="configs/${cfg}.yaml"
batch_size=16
epochs=50


lr=0.001
save_model="saved_model/${cfg}_${lr}_epoch${epochs}_batchsize${batch_size}_gpus${gpus}"

python ./pdseg/train.py --log_steps 10 --cfg ${config} --use_gpu --use_mpio \
--do_eval --use_tb --tb_log_dir ${save_model} \
BATCH_SIZE ${batch_size}  \
TRAIN.MODEL_SAVE_DIR ${save_model} \
SOLVER.LR ${lr} \
SOLVER.NUM_EPOCHS ${epochs}

lr=0.01
save_model="saved_model/${cfg}_${lr}_epoch${epochs}_batchsize${batch_size}_gpus${gpus}"

python ./pdseg/train.py --log_steps 10 --cfg ${config} --use_gpu --use_mpio \
--do_eval --use_tb --tb_log_dir ${save_model} \
BATCH_SIZE ${batch_size}  \
TRAIN.MODEL_SAVE_DIR ${save_model} \
SOLVER.LR ${lr} \
SOLVER.NUM_EPOCHS ${epochs}


lr=0.0001
save_model="saved_model/${cfg}_${lr}_epoch${epochs}_batchsize${batch_size}_gpus${gpus}"

python ./pdseg/train.py --log_steps 10 --cfg ${config} --use_gpu --use_mpio \
--do_eval --use_tb --tb_log_dir ${save_model} \
BATCH_SIZE ${batch_size}  \
TRAIN.MODEL_SAVE_DIR ${save_model} \
SOLVER.LR ${lr} \
SOLVER.NUM_EPOCHS ${epochs}


lr=0.1
save_model="saved_model/${cfg}_${lr}_epoch${epochs}_batchsize${batch_size}_gpus${gpus}"

python ./pdseg/train.py --log_steps 10 --cfg ${config} --use_gpu --use_mpio \
--do_eval --use_tb --tb_log_dir ${save_model} \
BATCH_SIZE ${batch_size}  \
TRAIN.MODEL_SAVE_DIR ${save_model} \
SOLVER.LR ${lr} \
SOLVER.NUM_EPOCHS ${epochs}

# sleep 10

# python ./pdseg/eval.py --cfg configs/land_cover3.yaml --use_gpu --use_mpio \
# TEST.TEST_MODEL ./saved_model/land_cover3_deeplabv3p_xception65_bn_cityscapes/500 \
# BATCH_SIZE 1




# python ./pdseg/vis.py --cfg ${config} --use_gpu  --vis_dir no_equibatch2 \
# TEST.TEST_MODEL "${save_model}/final" \
# BATCH_SIZE 4 

# cd visual/raw_results
# mkdir 450
# mv *.png 450
# cd ~/PaddleSeg


# --also_save_raw_results \
# DATASET.TEST_FILE_LIST "./dataset/land_cover/val.txt" \
# vis_file_list="dataset/puzhou2100_300/file_list/val.txt"
# #vis_file_list="None"
# resume_model_dir="${save_model}/20/"
# resume_model_dir="None"
