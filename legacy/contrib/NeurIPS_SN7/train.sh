source activate solaris
train_data_path=$1

rm -r /wdata/saved_model/hrnet/best_model/

rm -r /wdata/train
cp -r $train_data_path /wdata/train
rm /wdata/train/*

python tools.py /wdata/train train 2>err.log

cd pretrained_model
python download_model.py hrnet_w48_bn_imagenet
cd ..

python pdseg/train.py --do_eval --use_gpu --cfg hrnet_sn7.yaml DATASET.DATA_DIR /wdata/train DATASET.TEST_FILE_LIST val_list.txt
