>> 最好模型地址：链接：https://pan.baidu.com/s/1gPuqmhG6K2vo2eyBchEuqg 提取码：CAVS 

>> 预训练模型地址：链接：https://pan.baidu.com/s/1zDcUb4GxHH1xweRu3l2M9A 提取码：CAVS 
>> 预训练模型保存在PaddleSeg/pretrained中


数据集格式 
|-cityscapes    
 &#x2003|-leftImg8bit     
 &#x2003&#x2003|-train     
 &#x2003&#x2003|-val    
 &#x2003|-gtFine    
 &#x2003 &#x2003|-train    
 &#x2003 &#x2003|-val   
      
  ~~~Python
  #训练模型
  cd PaddleSeg
  Python main.py #单卡
  Python -m paddle.distributed.launch main.py  #多卡
  
  #验证模型
  cd PaddleSeg
  python valide.py
  ~~~
