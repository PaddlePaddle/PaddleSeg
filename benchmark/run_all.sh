# Test training benchmark for several models.

# Use docker： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37

# Usage:
#   git clone git clone https://github.com/PaddlePaddle/PaddleSeg.git
#   cd PaddleSeg
#   bash benchmark/run_all.sh

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
num_workers=5           # num_workers for dataloader

for model_name in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name} ${num_workers}
            sleep 60

            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name} ${num_workers}
            sleep 60
            done
      done
done

rm -rf data/*
# rm -rf ~/.paddleseg
