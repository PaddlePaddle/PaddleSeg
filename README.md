>> 最好模型地址：链接：https://pan.baidu.com/s/1gPuqmhG6K2vo2eyBchEuqg 提取码：CAVS   
>> 提取后放到PaddleSeg/output目录下

>> 预训练模型地址：链接：https://pan.baidu.com/s/1zDcUb4GxHH1xweRu3l2M9A 提取码：CAVS    
>> 预训练模型保存在PaddleSeg/pretrained中    

## 第一步克隆本项目：  
~~~shell
git clone https://github.com/marshall-dteach/PSA.git
cd PaddleSeg
~~~
## 第二步数据集格式
数据集格式    
|-cityscapes    
>|-leftImg8bit     
>>|-train     
>>|-val  

>|-gtFine    
>>|-train    
>>|-val   
 ## 第三步训练模型和模型预测
  #训练模型
  ~~~shell
  cd PaddleSeg
  python main.py #单卡
  python -m paddle.distributed.launch main.py  #多卡
  ~~~
  #验证模型
  ~~~shell
  cd PaddleSeg
  python valide.py
  ~~~
### 注意优化器选取根据单卡和四卡不同而调整
![image](https://user-images.githubusercontent.com/63546191/169228623-87f12422-54a4-449d-b42d-7e4d9baa22b0.png)
注意：单卡训练使用上面的lr，四卡训练使用下面的lr    

      
   
## 第四步模型精度检验
![image](https://user-images.githubusercontent.com/63546191/169260254-0892fcf1-1408-4bbf-9049-6dd7a3a5a88b.png)   
精度要求miou达到86.7，我的模型精度为86.87，满足要求

## 动态模型验证     
比较模型预测与ground truth    
![1](https://user-images.githubusercontent.com/63546191/169755335-068bbf51-25c2-4bc3-a589-adcc5c2261eb.png)  
![2](https://user-images.githubusercontent.com/63546191/169755356-e49bd5d2-b293-467f-8822-c40e959536e7.png)    
![3](https://user-images.githubusercontent.com/63546191/169755371-fe093a13-7115-4b86-9faf-1104c8c4c8b0.png)   
![4](https://user-images.githubusercontent.com/63546191/169755407-3fb01395-ec1d-4398-bfc8-20d42ce3950b.png)      
![5](https://user-images.githubusercontent.com/63546191/169755436-936867a7-d53f-4588-9b48-72fff455dc70.png)     
![6](https://user-images.githubusercontent.com/63546191/169755571-93992eb7-2a6e-4e3f-aa5f-10105d45f505.png)
  








