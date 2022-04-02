# PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model

## Reference

> xxx

## Performance

The implementation details refer to the config file.

### Cityscapes

| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PP-LiteSeg-T|STDC1|160000|1024x512|1025x512|73.10%|73.89%|-|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-T|STDC1|160000|1024x512|1536x768|76.03%|76.74%|-|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-T|STDC1|160000|1024x512|2048x1024|77.04%|77.73%|-|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-B|STDC2|160000|1024x512|1024x512|75.25%|75.65%|-|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-B|STDC2|160000|1024x512|1536x768|78.75%|79.23%|-|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-B|STDC2|160000|1024x512|2048x1024|79.04%|79.52%|-|[model]()\|[log]()\|[vdl]()|


### CamVid

| Model | Backbone | Training Iters | Train Resolution | Test Resolution | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|-|-|-|-|-|-|-|-|
|PP-LiteSeg-T|STDC1Net|10000|960x720|960x720|%|%|%|[model]()\|[log]()\|[vdl]()|
|PP-LiteSeg-B|STDC2Net|10000|960x720|960x720|%|%|%|[model]()\|[log]()\|[vdl]()|
