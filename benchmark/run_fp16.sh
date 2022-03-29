export FLAGS_conv_workspace_size_limit=2000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python train.py --config benchmark/deeplabv3p.yml \
  --iters=500 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --num_workers 8 \
  --log_iters 20 \
  --data_format NHWC \
  --precision fp16
