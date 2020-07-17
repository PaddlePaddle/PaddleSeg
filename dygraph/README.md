# 动态图执行

## 训练
```
python3 train.py --model_name UNet \
--dataset OpticDiscSeg \
--input_size 192 192 \
--num_epochs 10 \
--save_interval_epochs 1 \
--do_eval \
--save_dir output
```

## 评估
```
python3 val.py --model_name UNet \
--dataset OpticDiscSeg \
--input_size 192 192 \
--model_dir output/best_model
```

## 预测
```
python3 infer.py --model_name UNet \
--dataset OpticDiscSeg \
--model_dir output/best_model \
--input_size 192 192
```
