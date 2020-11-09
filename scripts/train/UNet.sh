#UNet

export CUDA_VISIBLE_DEVICES=4

DATA_FORMAT="NCHW"
# DATA_FORMAT="NHWC"

python3.6 pdseg/train.py --cfg configs/unet_optic.yaml \
        --use_gpu \
        --data_format=${DATA_FORMAT} \
        BATCH_SIZE 8 \
        SOLVER.LR 0.001 \
        SOLVER.OPTIMIZER "sgd"
