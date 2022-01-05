# Test training benchmark for several models.

# Use docker： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37

# Usage:
#   git clone git clone https://github.com/PaddlePaddle/PaddleSeg.git
#   cd PaddleSeg
#   bash benchmark/run_all.sh

#if  [ ${RUN_PROFILER} = "PROFILER" ]; then
#    log_path=${PROFILER_LOG_DIR:-$(pwd)}  #  benchmark系统指定该参数,如果需要跑profile时,log_path指向存profile的目录
#fi
log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}  #  benchmark系统指定该参数,不需要跑profile时,log_path指向存speed的目录

pip install -r requirements.txt
# Download test dataset and save it to PaddleSeg/data
# It automatic downloads the pretrained models saved in ~/.paddleseg
mkdir -p data
wget https://paddleseg.bj.bcebos.com/dataset/cityscapes_30imgs.tar.gz \
    -O data/cityscapes_30imgs.tar.gz
tar -zxf data/cityscapes_30imgs.tar.gz -C data/

model_name_list=(fastscnn segformer_b0 ocrnet_hrnetw48)
fp_item_list=(fp32)     # set fp32 or fp16, segformer_b0 doesn't support fp16 with Paddle2.1.2
bs_list=(2)
max_iters=500           # control the test time
num_workers=5           # num_workers for dataloader, as for fastscnn and ocrnet_hrnetw48, it is better to set 8

for model_name in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            run_mode=sp
            log_name=seg_${model_name}_${run_mode}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
            echo "index is speed, 1gpus, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name} ${num_workers} | tee ${log_path}/${log_name}_speed_1gpus 2>&1
            sleep 60

            run_mode=mp
            log_name=seg_${model_name}_${run_mode}_bs${bs_item}_${fp_item}
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name} ${num_workers} | tee ${log_path}/${log_name}_speed_8gpus8p 2>&1
            sleep 60
            done
      done
done

rm -rf data/*
# rm -rf ~/.paddleseg
