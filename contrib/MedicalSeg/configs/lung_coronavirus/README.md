# [COVID-19 CT scans](https://www.kaggle.com/andrewmvd/covid19-ct-scans)
20 CT scans and expert segmentations of patients with COVID-19

## Performance

### Vnet
> Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for volumetric medical image segmentation." In 2016 fourth international conference on 3D vision (3DV), pp. 565-571. IEEE, 2016.

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|128x128x128|0.001|15000|97.04%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_1e-3/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=9db5c1e11ebc82f9a470f01a9114bd3c)|
|-|128x128x128|0.0003|15000|92.70%|[model](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/lung_coronavirus/vnet_lung_coronavirus_128_128_128_15k_3e-4/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=0fb90ee5a6ea8821c0d61a6857ba4614)|


### Unet
> Çiçek, Özgün, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. "3D U-Net: learning dense volumetric segmentation from sparse annotation." In International conference on medical image computing and computer-assisted intervention, pp. 424-432. Springer, Cham, 2016.

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|

To be continue.
