# [MRISpineSeg](https://www.spinesegmentation-challenge.com/)
There are 172 training data in the preliminary competition, including MR images and mask labels, 20 test data in the preliminary competition and 23 test data in the  second round competition. The labels of the preliminary competition testset and the second round competition testset are not published.

## Performance

### Vnet
> Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for volumetric medical image segmentation." In 2016 fourth international conference on 3D vision (3DV), pp. 565-571. IEEE, 2016.

| Backbone | Resolution | lr | Training Iters | Dice(20 classes) | Dice(16 classes*) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|-|512x512x12|0.1|15000|74.41%| 88.17% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_1e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=36504064c740e28506f991815bd21cc7)|
|-|512x512x12|0.5|15000|74.69%| 89.14% |[model](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/paddleseg3d/mri_spine_seg/vnet_mri_spine_seg_512_512_12_15k_5e-1/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=08b0f9f62ebb255cdfc93fd6bd8f2c06)|

16 classes*: 16 classes removed T9, T10, T9/T10 and T10/T11 from calculating the mean Dice compared from the 20 classes.



### Unet
> Çiçek, Özgün, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. "3D U-Net: learning dense volumetric segmentation from sparse annotation." In International conference on medical image computing and computer-assisted intervention, pp. 424-432. Springer, Cham, 2016.

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|

To be continue.
