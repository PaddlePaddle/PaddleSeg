#!/bin/sh
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98


python -u ./slim/quantization/train_quant.py --log_steps 10 --not_quant_pattern last_conv --cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml --use_gpu --use_mpio --do_eval \
TRAIN.PRETRAINED_MODEL_DIR "./pretrain/mobilenet_cityscapes/" \
TRAIN.MODEL_SAVE_DIR "./snapshots/mobilenetv2_quant" \
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
SOLVER.LR 0.0001 \
TRAIN.SNAPSHOT_EPOCH 1 \
SOLVER.NUM_EPOCHS 30 \
SOLVER.LR_POLICY "poly" \
SOLVER.OPTIMIZER "sgd" \
BATCH_SIZE 16 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.BUF_SIZE 256
