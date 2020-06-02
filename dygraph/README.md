# 动态图执行

## 数据集设置
```
data_dir='data/path'
train_list='train/list/path'
val_list='val/list/path'
test_list='test/list/path'
num_classes=number/of/dataset/classes
```

## 训练
```
python3 train.py --model_name UNet \
--data_dir $data_dir \
--train_list $train_list \
--val_list $val_list \
--num_classes $num_classes \
--input_size 192 192 \
--num_epochs 4 \
--save_interval_epochs 1 \
--save_dir output
```

## 评估
```
python3 val.py --model_name UNet \
--data_dir $data_dir \
--val_list $val_list \
--num_classes $num_classes \
--input_size 192 192 \
--model_dir output/epoch_1
```

## 预测
```
python3 infer.py --model_name UNet \
--data_dir $data_dir \
--test_list $test_list \
--num_classes $num_classes \
--input_size 192 192 \
--model_dir output/epoch_1
```
