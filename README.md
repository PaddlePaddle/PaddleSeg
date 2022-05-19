>> 最好模型地址：链接：https://pan.baidu.com/s/1gPuqmhG6K2vo2eyBchEuqg 提取码：CAVS 

>> 预训练模型地址：链接：https://pan.baidu.com/s/1zDcUb4GxHH1xweRu3l2M9A 提取码：CAVS 
>> 预训练模型保存在PaddleSeg/pretrained中


数据集格式    
|-cityscapes    
>|-leftImg8bit     
>>|-train     
>>|-val  

>|-gtFine    
>>|-train    
>>|-val   
      
  ~~~Python
  #训练模型
  cd PaddleSeg
  python main.py #单卡
  python -m paddle.distributed.launch main.py  #多卡
  
  #验证模型
  cd PaddleSeg
  python valide.py
  ~~~

![image](https://user-images.githubusercontent.com/63546191/169228623-87f12422-54a4-449d-b42d-7e4d9baa22b0.png)
注意：单卡训练使用上面的lr，四卡训练使用下面的lr
