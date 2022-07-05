# 执行位置在该模型套件的根目录
model_item=HRnet
bs_item=8
fp_item=fp32
run_mode=DP
device_num=N1C8
# get data
#bash test_tipc/static/${model_item}/benchmark_common/prepare.sh cityscapes
# run 
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
