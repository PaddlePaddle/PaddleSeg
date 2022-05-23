# download data
mkdir data
cd data
wget https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip
unzip optic_disc_seg.zip > /dev/null
cd ..

# check traing
python train.py \
    --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
    --iters 20 \
    --do_eval \
    --save_interval 20 \
    --save_dir output/pp_liteseg_optic_disc

echo -e "\n\n"

# check predicting
python predict.py \
    --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
    --model_path https://paddleseg.bj.bcebos.com/dygraph/optic_disc/pp_liteseg_optic_disc_512x512_1k/model.pdparams\
    --image_path docs/images/optic_test_image.jpg \
    --save_dir output/result
