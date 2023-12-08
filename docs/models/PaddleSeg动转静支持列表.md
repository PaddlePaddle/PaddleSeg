## PaddleSeg动转静模型支持列表

|     模型名称      | 是否支持 |
| :---------------: | :------: |
|        ann        |    ✅     |
|  attention_unet   |      ✅     |
|      bisenet      |   ✅     |
|     bisenetv1     |     ✅     |
|        cae        |    静态图出现错误       |
|       ccnet       |     ✅     |
|       danet       |     ✅     |
|      ddrnet       |      ✅     |
| decoupled_segnet  |      ✅     |
|     deeplabv3     |    ✅     |
|    deeplabv3p     |     ✅     |
|       dmnet       |   ✅     |
|      dnlnet       |  ✅     |
| efficientformerv2 |   ✅     |
|      emanet       |   静态图出现错误      |
|      encnet       |    ✅     |
|       enet        |     ✅     |
|      emanet       |      ✅     |
|      encnet       |  ✅     |
|       enet        |  ✅     |
|      espnet       |  ✅     |
|      fastfcn      |    ✅     |
|     fastscnn      |    ✅     |
|        fcn        |  ✅     |
|       gcnet       |     ✅     |
|       ginet       |    ✅     |
|       glore       |    ✅     |
|       gscnn       |  静态图出现错误        |
|      hardnet      |    ✅     |
|     hrformer      |     ✅     |
|       hrnet       |    ✅     |
|      isanet       |     ✅     |
|       knet        |     ✅     |
|      lraspp       |     ✅     |
|     mobileseg     |       ✅     |
|   multilabelseg   |       ✅     |
|      ocrnet       |      ✅     |
|       pfpn        |      ✅     |
|     pointrend     |    ✅     |
|    portraitnet    |      ✅     |
|    pp_humanseg    |     ✅     |
|    pp_liteseg     |       ✅     |
|   pp_mobileseg    |      ✅     |
|      pspnet       |      ✅     |
|     rtformer      |      ✅     |
|     seaformer     |    ✅     |
|     segformer     |      ✅     |
|     segmenter     |     ✅     |
|      segnet       |      ✅     |
|      segnext      |       ✅     |
|       setr        |    静态图发生错误    |
|       sfnet       |   ✅     |
|       smrt        |   ✅     |
|      stdcseg      |  ✅     |
|     topformer     |      ✅     |
|       u2net       |     ✅     |
|      uhrnet       |     ✅     |
|       unet        |     ✅     |
|    unet_3plus     |     ✅     |
|   unet_plusplus   |    ✅     |
|      upernet      |       ✅     |


## 运行命令
```shell
python tools/train.py --config configs/attention_unet/attention_unet_cityscapes_1024x512_80k.yml  --save_interval 500 --do_eval  --use_vdl --save_dir output --seed 1220
```
