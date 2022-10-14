# PP-Matting: High-Accuracy Natural Image Matting

## Reference

> Chen G, Liu Y, Wang J, et al. PP-Matting: High-Accuracy Natural Image Matting[J]. arXiv preprint arXiv:2204.09433, 2022.

## Performance

### Composition-1k

| Model | Backbone | Resolution | Training Iters | SAD $\downarrow$ | MSE $\downarrow$ | Grad $\downarrow$ | Conn $\downarrow$ | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-Matting|HRNet_W48|512x512|300000|46.22|0.005|22.69|45.40|[model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w48-composition.pdparams)|


### Distinctions-646

| Model | Backbone | Resolution | Training Iters | SAD $\downarrow$ | MSE $\downarrow$ | Grad $\downarrow$ | Conn $\downarrow$ | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PP-Matting|HRNet_W48|512x512|300000|40.69|0.009|43.91|40.56|[model](https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w48-distinctions.pdparams)|
