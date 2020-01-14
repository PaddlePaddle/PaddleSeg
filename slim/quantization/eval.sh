#!/bin/sh
#export CUDA_VISIBLE_DEVICES=3
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
#export GLOG_vmodule=operator=5
#export GLOG_v=10


python -u ./slim/quantization/eval_quant.py  --cfg configs/cityscape.yaml --use_gpu --not_quant_pattern last_conv  --use_mpio --convert \
DATASET.SEPARATOR " " \
TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" \
MODEL.DEEPLAB.BACKBONE "mobilenet" \
MODEL.FP16 False \
MODEL.SCALE_LOSS "dynamic" \
MODEL.ICNET.DEPTH_MULTIPLIER 1.0 \
MODEL.DEEPLAB.ASPP_WITH_SEP_CONV True \
MODEL.DEEPLAB.DECODER_USE_SEP_CONV True \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
MODEL.DEFAULT_NORM_TYPE "bn" \
TRAIN.SYNC_BATCH_NORM False \
BATCH_SIZE 16 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.BUF_SIZE 256 \
