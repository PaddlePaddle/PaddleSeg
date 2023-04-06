# check predicting
python tools/predict.py \
    --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
    --model_path https://paddleseg.bj.bcebos.com/dygraph/optic_disc/pp_liteseg_optic_disc_512x512_1k/model.pdparams \
    --image_path docs/images/optic_test_image.jpg \
    --save_dir output/result
