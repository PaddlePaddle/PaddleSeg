# 介绍文件
>> 最好模型地址：链接：https://pan.baidu.com/s/1u3gOV6nhswt_U3X1pDNx7Q 提取码：CAVS   
>> 提取后放到PaddleSeg/output目录下

>> 预训练模型地址：链接：https://pan.baidu.com/s/1q9Rnf1_hmBlqkKYQofoNhg 提取码：CAVS    
>> 预训练模型保存在PaddleSeg/pretrained中    

## 第一步克隆本项目：  
~~~shell
git clone https://github.com/marshall-dteach/PSA.git
~~~
## 第二步数据集格式
数据集格式    
```css
.
└── cityscapes
    ├── leftImg8bit
    │   └── train
	│		└── acchen
	│	└── val
	│		└── acchen
    └── gtFine 
        ├── train
        ├── val
```

 ## 第三步训练模型和模型预测
  #训练模型
  ~~~shell
python train.py --config configs/psa/psa_cityscapes_1024x2048_520k.yml \
python -m paddle.distributed.launch train.py \
		--config configs/psa/psa_cityscapes_1024x2048_520k.yml \
  ~~~
  #验证模型
  ~~~shell
python val.py \
        --config configs/psa/psa_cityscapes_1024x2048_520k.yml \
        --model_path ../model.pdparams \

python val.py \
        --config configs/psa/psa_cityscapes_1024x2048_520k.yml \
        --model_path output/model.pdparams \
        --aug_eval \
        --flip_horizontal
        
python val.py \
        --config configs/psa/psa_cityscapes_1024x2048_520k.yml \
        --model_path output/model.pdparams \
        --aug_eval \
        --scale 0.75 1.0 1.25 \
        --flip_horizontal
  ~~~
​      

## 第四步模型精度检验
![image](https://user-images.githubusercontent.com/63546191/169260254-0892fcf1-1408-4bbf-9049-6dd7a3a5a88b.png)   
精度要求miou达到86.7，我的模型精度为86.87，满足要求

# TIPC

~~~shell
bash test_tipc/prepare.sh ./test_tipc/configs/psa/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/psa/train_infer_python.txt 'lite_train_lite_infer'
~~~



## 动态模型验证     
比较模型预测与ground truth    
![1](https://user-images.githubusercontent.com/63546191/169755335-068bbf51-25c2-4bc3-a589-adcc5c2261eb.png)  
![2](https://user-images.githubusercontent.com/63546191/169755356-e49bd5d2-b293-467f-8822-c40e959536e7.png)    
![3](https://user-images.githubusercontent.com/63546191/169755371-fe093a13-7115-4b86-9faf-1104c8c4c8b0.png)   
![4](https://user-images.githubusercontent.com/63546191/169755407-3fb01395-ec1d-4398-bfc8-20d42ce3950b.png)      
![5](https://user-images.githubusercontent.com/63546191/169755436-936867a7-d53f-4588-9b48-72fff455dc70.png)     
![6](https://user-images.githubusercontent.com/63546191/169755571-93992eb7-2a6e-4e3f-aa5f-10105d45f505.png)   