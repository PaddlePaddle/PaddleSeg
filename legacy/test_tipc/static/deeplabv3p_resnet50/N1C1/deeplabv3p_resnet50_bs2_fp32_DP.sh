# 执行位置在该模型套件的根目录
model_item=deeplabv3p_resnet50
bs_item=2
fp_item=fp32
run_mode=DP
device_num=N1C1
# get data
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh cityscapes
# run 
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
# run profiling  skip