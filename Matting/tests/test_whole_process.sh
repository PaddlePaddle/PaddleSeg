# This file tests the whole process from training to deployment

# Usage:
#   1. Install PaddlePaddle that supports TenorRT
#   2. `export CUDA_VISIBLE_DEVICES=id`
#   3. `cd ./PaddleSeg/Matting`
#.  4. `bash tests/test_whole_process.sh`


save_root="output/tests"
mkdir -p ${save_root}
video_path="${save_root}/v1.mov"

# Obtain dataset to test
dataset_root="data/PPM-100"
if [ ! -d ${dataset_root} ]; then
    mkdir -p data && cd data
    wget https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip
    unzip PPM-100.zip
    rm PPM-100.zip
    cd ..
fi
# Obtaion video to test
if [ ! -a ${video_path} ]; then
    wget https://paddleseg.bj.bcebos.com/matting/demo/v1.mov
    mv v1.mov ${save_root}
fi

# Training
echo "Test training..."
python tools/train.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --learning_rate 0.0001 \
    --iters 10 \
    --batch_size 1 \
    --log_iters 1 \
    --use_vdl \
    --save_interval 10 \
    --do_eval \
    --num_workers 1 \
    --save_dir ${save_root} \
    --opts model.pretrained="https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams"

# Evaluation
echo "Test evaluation..."
python tools/val.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --metrics sad mse grad conn \
    --save_dir ${save_root}/results/evaluation \
    --save_results

# Predictions
echo "Test prediction..."
python tools/predict.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --image_path demo/human.jpg \
    --save_dir ${save_root}/results/prediction \
    --fg_estimate True

# Video prediction
echo "Test video predcition..."
python tools/predict_video.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --video_path ${video_path} \
    --save_dir ${save_root}/results/video_prediction \
    --fg_estimate False

# Background replacement
echo "Test background replacement..."
python tools/bg_replace.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --image_path demo/human.jpg \
    --background g \
    --save_dir ${save_root}/results/background_replacement \
    --fg_estimate True

# Video background replacement
echo "Test video background replacement..."
python tools/bg_replace_video.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --video_path ${video_path} \
    --background 'g' \
    --save_dir ${save_root}/results/video_background_replacement \
    --fg_estimate False

# Export
echo "Test exportment..."
python tools/export.py \
    --config configs/quick_start/ppmattingv2-stdc1-human_512.yml \
    --model_path ${save_root}/best_model/model.pdparams \
    --save_dir ${save_root}/export \
    --input_shape 1 3 512 512

# Deployment
echo "Test deployment"
python deploy/python/infer.py \
    --config ${save_root}/export/deploy.yaml \
    --image_path demo/human.jpg \
    --save_dir ${save_root}/results/deploy \

python deploy/python/infer.py \
    --config ${save_root}/export/deploy.yaml \
    --video_path ${video_path} \
    --save_dir ${save_root}/results/deploy \
    --fg_estimate False
