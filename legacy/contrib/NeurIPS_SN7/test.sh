source activate solaris
test_data_path=$1
output_path=$2

if [ ! -d /wdata/saved_model/hrnet/best_model ]; then
    bash download.sh
fi

rm -r /wdata/test
cp -r $test_data_path /wdata/test
rm /wdata/test/*

python tools.py /wdata/test test
cp dummy.tif /wdata/test

python pdseg/eval.py --use_gpu --vis --vis_dir vis/test_org --cfg hrnet_sn7.yaml DATASET.DATA_DIR /wdata/test DATASET.VAL_FILE_LIST test_list.txt VIS.VISINEVAL True TEST.TEST_AUG True

python tools.py vis/test_org compose

python postprocess.py /wdata/test vis/test_org_compose "$output_path"

rm -r vis
