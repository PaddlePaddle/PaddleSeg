# U-Net

U-Net [1] originated from medical image segmentation, and has the characteristics of few parameters, fast calculation, strong applicability, and high adaptability to general scenes. U-Net was first proposed in 2015 and won the first place in the ISBI 2015 Cell Tracking Challenge. After development, there are currently several variants and applications. The structure of the original U-Net is a standard encoder-decoder structure. As shown in the figure below, the left side can be regarded as an encoder, and the right side can be regarded as a decoder. The encoder consists of four sub-modules, each sub-module contains two convolutional layers, and each sub-module is then downsampled by the max pool. The encoder as a whole presents a gradually shrinking structure, continuously reducing the spatial dimension of the pooling layer and reducing the resolution of the feature map to capture contextual information.
The decoder presents a symmetrical expansion structure with the encoder, and gradually repairs the details and spatial dimensions of the segmented objects to achieve precise positioning. The decoder also includes four sub-modules, and the resolution is sequentially increased through the upsampling operation until it is basically the same as the resolution of the input image.
The network also uses skip connections, that is, every time the decoder upsamples, the feature maps corresponding to the same resolution in the decoder and encoder are fused in a spliced ​​manner to help the decoder better recover the details of the target. Because the overall structure of the network is similar to the capital letter U, it is named U-Net.

![](./images/UNet.png)

<div align = "center">U-Net</div>

For details, please refer to[U-Net:Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
