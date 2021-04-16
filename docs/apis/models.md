# paddleseg.models

The models subpackage contains the following model for image sementic segmentaion.
- [DeepLabV3+](#DeepLabV3)
- [DeepLabV3](#DeepLabV3-1)
- [FCN](#FCN)
- [OCRNet](#OCRNet)
- [PSPNet](#PSPNet)
- [ANN](#ANN)
- [BiSeNetV2](#BiSeNetV2)
- [DANet](#DANet)
- [FastSCNN](#FastSCNN)
- [GCNet](#GCNet)
- [GSCNN](#GSCNN)
- [HarDNet](#HarDNet)
- [UNet](#UNet)
- [U<sup>2</sup>Net](#U2Net)
- [U<sup>2</sup>Net+](#U2Net-1)
- [AttentionUNet](#AttentionUNet)
- [UNet++](#UNet-1)
- [UNet3+](#UNet-2)
- [DecoupledSegNet](#DecoupledSegNet)
- [ISANet](#ISANet)
- [EMANet](#EMANet)
- [DNLNet](#DNLNet)


## [DeepLabV3+](../../paddleseg/models/deeplab.py)
> CLASS paddleseg.models.DeepLabV3P(num_classes, backbone, backbone_indices=(0, 3), aspp_ratios=(1, 6, 12, 18), aspp_out_channels=256, align_corners=False, pretrained=None)

    The DeepLabV3Plus implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
       Default: (0, 3).
> > > - **aspp_ratios** (tuple, optional): The dilation rate using in ASSP module.
        If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
        If output_stride=8, aspp_ratios is (1, 12, 24, 36).
        Default: (1, 6, 12, 18).
> > > - **aspp_out_channels** (int, optional): The output channels of ASPP module. Default: 256.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
        e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [DeepLabV3](../../paddleseg/models/deeplab.py)
> CLASS paddleseg.models.DeepLabV3(num_classes, backbone, backbone_indices=(3, ), aspp_ratios=(1, 6, 12, 18), aspp_out_channels=256, align_corners=False, pretrained=None)

    The DeepLabV3 implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
       Default: (3, ).
> > > - **aspp_ratios** (tuple, optional): The dilation rate using in ASSP module.
        If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
        If output_stride=8, aspp_ratios is (1, 12, 24, 36).
        Default: (1, 6, 12, 18).
> > > - **aspp_out_channels** (int, optional): The output channels of ASPP module. Default: 256.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
        e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [FCN](../../paddleseg/models/deeplab.py)
> CLASS paddleseg.models.FCN(num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None)

    A simple implementation for FCN based on PaddlePaddle.

    The original article refers to
    Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1411.4038).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (paddle.nn.Layer): Backbone networks.
> > > - **backbone_indices** (tuple, optional): The values in the tuple indicate the indices of output of backbone.
        Default: (-1, ).
> > > - **channels** (int, optional): The channels between conv layer and the last layer of FCNHead.
        If None, it will be the number of channels of input features. Default: None.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
        is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None

## [OCRNet](../../paddleseg/models/ocrnet.py)
> CLASS paddleseg.models.OCRNet(num_classes,
                 backbone,
                 backbone_indices,
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 align_corners=False,
                 pretrained=None)

    The OCRNet implementation based on PaddlePaddle.
    The original article refers to
        Yuan, Yuhui, et al. "Object-Contextual Representations for Semantic Segmentation"
        (https://arxiv.org/pdf/1909.11065.pdf)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): Backbone network.
> > > - **backbone_indices** (tuple): A tuple indicates the indices of output of backbone.
            It can be either one or two values, if two values, the first index will be taken as
            a deep-supervision feature in auxiliary layer; the second one will be taken as
            input of pixel representation. If one value, it is taken by both above.
> > > - **ocr_mid_channels** (int, optional): The number of middle channels in OCRHead. Default: 512.
> > > - **ocr_key_channels** (int, optional): The number of key channels in ObjectAttentionBlock. Default: 256.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [PSPNet](../../paddleseg/models/pspnet.py)
> CLASS paddleseg.models.PSPNet(num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None)

    The PSPNet implementation based on PaddlePaddle.

    The original article refers to
    Zhao, Hengshuang, et al. "Pyramid scene parsing network"
    (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
> > > - **pp_out_channels** (int, optional): The output channels after Pyramid Pooling Module. Default: 1024.
> > > - **bin_sizes** (tuple, optional): The out size of pooled feature maps. Default: (1,2,3,6).
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [ANN](../../paddleseg/models/ann.py)
> CLASS paddleseg.models.ANN(num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 key_value_channels=256,
                 inter_channels=512,
                 psp_size=(1, 3, 6, 8),
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None)

    The ANN implementation based on PaddlePaddle.

    The original article refers to
    Zhen, Zhu, et al. "Asymmetric Non-local Neural Networks for Semantic Segmentation"
    (https://arxiv.org/pdf/1908.07678.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
> > > - **key_value_channels** (int, optional): The key and value channels of self-attention map in both AFNB and APNB modules.
            Default: 256.
> > > - **inter_channels** (int, optional): Both input and output channels of APNB modules. Default: 512.
> > > - **psp_size** (tuple, optional): The out size of pooled feature maps. Default: (1, 3, 6, 8).
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [BiSeNetV2](../../paddleseg/models/bisenet.py)
> CLASS paddleseg.models.BiSeNetV2(num_classes,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)

    The BiSeNet V2 implementation based on PaddlePaddle.

    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://arxiv.org/abs/2004.02147)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **lambd** (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [DANet](../../paddleseg/models/danet.py)
> CLASS paddleseg.models.DANet(num_classes,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)

    The DANet implementation based on PaddlePaddle.

    The original article refers to
    Fu, jun, et al. "Dual Attention Network for Scene Segmentation"
    (https://arxiv.org/pdf/1809.02983.pdf)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): A backbone network.
> > > - **backbone_indices** (tuple): The values in the tuple indicate the indices of
            output of backbone.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [FastSCNN](../../paddleseg/models/fast_scnn.py)
> CLASS paddleseg.models.FastSCNN(num_classes,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None)

    The FastSCNN implementation based on PaddlePaddle.
    As mentioned in the original paper, FastSCNN is a real-time segmentation algorithm (123.5fps)
    even for high resolution images (1024x2048).
    The original article refers to
    Poudel, Rudra PK, et al. "Fast-scnn: Fast semantic segmentation network"
    (https://arxiv.org/pdf/1902.04502.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss.
            If true, auxiliary loss will be added after LearningToDownsample module. Default: False.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [GCNet](../../paddleseg/models/gcnet.py)
> CLASS paddleseg.models.GCNet(num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 gc_channels=512,
                 ratio=0.25,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None)

    The GCNet implementation based on PaddlePaddle.

    The original article refers to
    Cao, Yue, et al. "GCnet: Non-local networks meet squeeze-excitation networks and beyond"
    (https://arxiv.org/pdf/1904.11492.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
> > > - **gc_channels** (int, optional): The input channels to Global Context Block. Default: 512.
> > > - **ratio** (float, optional): It indicates the ratio of attention channels and gc_channels. Default: 0.25.
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.


## [GSCNN](../../paddleseg/models/gscnn.py)
> CLASS paddleseg.models.GSCNN(num_classes,
                 backbone,
                 backbone_indices=(0, 1, 2, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None)

    The GSCNN implementation based on PaddlePaddle.

    The original article refers to
    Towaki Takikawa, et, al. "Gated-SCNN: Gated Shape CNNs for Semantic Segmentation"
    (https://arxiv.org/pdf/1907.05740.pdf)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 1, 2, 3).
> > > - **aspp_ratios** (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
> > > - **aspp_out_channels** (int, optional): The output channels of ASPP module. Default: 256.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [HarDNet](../../paddleseg/models/hardnet.py)
> CLASS paddleseg.models.HarDNet(num_classes,
                 stem_channels=(16, 24, 32, 48),
                 ch_list=(64, 96, 160, 224, 320),
                 grmul=1.7,
                 gr=(10, 16, 18, 24, 32),
                 n_layers=(4, 4, 8, 8, 8),
                 align_corners=False,
                 pretrained=None)

    [Real Time] The FC-HardDNet 70 implementation based on PaddlePaddle.
    The original article refers to
        Chao, Ping, et al. "HarDNet: A Low Memory Traffic Network"
        (https://arxiv.org/pdf/1909.00948.pdf)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **stem_channels** (tuple|list, optional): The number of channels before the encoder. Default: (16, 24, 32, 48).
> > > - **ch_list** (tuple|list, optional): The number of channels at each block in the encoder. Default: (64, 96, 160, 224, 320).
> > > - **grmul** (float, optional): The channel multiplying factor in HarDBlock, which is m in the paper. Default: 1.7.
> > > - **gr** (tuple|list, optional): The growth rate in each HarDBlock, which is k in the paper. Default: (10, 16, 18, 24, 32).
> > > - **n_layers** (tuple|list, optional): The number of layers in each HarDBlock. Default: (4, 4, 8, 8, 8).
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [UNet](../../paddleseg/models/unet.py)
> CLASS paddleseg.models.UNet(num_classes,
                 align_corners=False,
                 use_deconv=False,
                 pretrained=None)

    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **use_deconv** (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model for fine tuning. Default: None.

## [U<sup>2</sup>Net](../../paddleseg/models/u2net.py)
> CLASS paddleseg.models.U2Net(num_classes, in_ch=3, pretrained=None)

    The U^2-Net implementation based on PaddlePaddle.

    The original article refers to
    Xuebin Qin, et, al. "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
    (https://arxiv.org/abs/2005.09007).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **in_ch** (int, optional): Input channels. Default: 3.
> > > - **pretrained** (str, optional): The path or url of pretrained model for fine tuning. Default: None.

## [U<sup>2</sup>Net+](../../paddleseg/models/u2net.py)
> CLASS paddleseg.models.U2Netp(num_classes, in_ch=3, pretrained=None)

    The U^2-Netp implementation based on PaddlePaddle.

    The original article refers to
    Xuebin Qin, et, al. "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
    (https://arxiv.org/abs/2005.09007).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **in_ch** (int, optional): Input channels. Default: 3.
> > > - **pretrained** (str, optional): The path or url of pretrained model for fine tuning. Default: None.

## [AttentionUNet](../../paddleseg/models/attention_unet.py)
> CLASS paddleseg.models.AttentionUNet(num_classes, pretrained=None)

    The Attention-UNet implementation based on PaddlePaddle.
    As mentioned in the original paper, author proposes a novel attention gate (AG)
    that automatically learns to focus on target structures of varying shapes and sizes.
    Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while
    highlighting salient features useful for a specific task.

    The original article refers to
    Oktay, O, et, al. "Attention u-net: Learning where to look for the pancreas."
    (https://arxiv.org/pdf/1804.03999.pdf).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [UNet++](../../paddleseg/models/unet_plusplus.py)
> class UNetPlusPlus(in_channels,
                 num_classes,
                 use_deconv=False,
                 align_corners=False,
                 pretrained=None,
                 is_ds=True)

    The UNet++ implementation based on PaddlePaddle.

    The original article refers to
    Zongwei Zhou, et, al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    (https://arxiv.org/abs/1807.10165).

> > Args
> > > - **in_channels** (int): The channel number of input image.
> > > - **num_classes** (int): The unique number of target classes.
> > > - **use_deconv** (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model for fine tuning. Default: None.
> > > - **is_ds** (bool): use deep supervision or not. Default: True

## <span id="UNet-2">[UNet3+](../../paddleseg/models/unet_3plus.py)</span>
> class UNet3Plus(in_channels,
                 num_classes,
                 is_batchnorm=True,
                 is_deepsup=False,
                 is_CGM=False)

    The UNet3+ implementation based on PaddlePaddle.

    The original article refers to
    Huang H , Lin L , Tong R , et al. "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation"
    (https://arxiv.org/abs/2004.08790).

> > Args
> > > - **in_channels** (int): The channel number of input image.
> > > - **num_classes** (int): The unique number of target classes.
> > > - **is_batchnorm** (bool, optional) Use batchnorm after conv or not.  Default: True.
> > > - **is_deepsup** (bool, optional): Use deep supervision or not.  Default: False.
> > > - **is_CGM** (bool, optional): Use classification-guided module or not.
            If True, is_deepsup must be True.  Default: False.

## [DecoupledSegNet](../../paddleseg/models/decoupled_segnet.py)
> class DecoupledSegNet(num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None)

    The DecoupledSegNet implementation based on PaddlePaddle.

    The original article refers to
    Xiangtai Li, et, al. "Improving Semantic Segmentation via Decoupled Body and Edge Supervision"
    (https://arxiv.org/pdf/2007.10035.pdf)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd.
> > > - **backbone_indices** (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
> > > - **aspp_ratios** (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
> > > - **aspp_out_channels** (int, optional): The output channels of ASPP module. Default: 256.
> > > - **align_corners** (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [ISANet](../../paddleseg/models/isanet.py)
> CLASS paddleseg.models.ISANet(num_classes, backbone, backbone_indices=(2, 3), isa_channels=256, down_factor=(8, 8), enable_auxiliary_loss=True, align_corners=False, pretrained=None)

    The ISANet implementation based on PaddlePaddle.

    The original article refers to Lang Huang, et al. "Interlaced Sparse Self-Attention for Semantic Segmentation"
    (https://arxiv.org/abs/1907.12273).

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): A backbone network.
> > > - **backbone_indices** (tuple): The values in the tuple indicate the indices of output of backbone.
> > > - **isa_channels** (int): The channels of ISA Module.
> > > - **down_factor** (tuple): Divide the height and width dimension to (Ph, PW) groups.
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [EMANet](../../paddleseg/models/emanet.py)
> CLASS paddleseg.models.EMANet(num_classes, backbone, backbone_indices=(2, 3), ema_channels=512, gc_channels=256, num_bases=64, stage_num=3, momentum=0.1, concat_input=True, enable_auxiliary_loss=True, align_corners=False, pretrained=None)

    The EMANet implementation based on PaddlePaddle.

    The original article refers to
    Xia Li, et al. "Expectation-Maximization Attention Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1907.13426)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): A backbone network.
> > > - **backbone_indices** (tuple): The values in the tuple indicate the indices of output of backbone.
> > > - **ema_channels** (int): EMA module channels.
> > > - **gc_channels** (int): The input channels to Global Context Block.
> > > - **num_bases** (int): Number of bases.
> > > - **stage_num** (int): The iteration number for EM.
> > > - **momentum** (float): The parameter for updating bases.
> > > - **concat_input** (bool): Whether concat the input and output of convs before classification layer. Default: True
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [DNLNet](../../paddleseg/models/dnlnet.py)
> CLASS paddleseg.models.DNLNet(num_classes, backbone, backbone_indices=(2, 3), reduction=2, use_scale=True, mode='embedded_gaussian', temperature=0.05, concat_input=True, enable_auxiliary_loss=True, align_corners=False, pretrained=None)

    The DNLNet implementation based on PaddlePaddle.

    The original article refers to
    Minghao Yin, et al. "Disentangled Non-Local Neural Networks"
    (https://arxiv.org/abs/2006.06668)

> > Args
> > > - **num_classes** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): A backbone network.
> > > - **backbone_indices** (tuple): The values in the tuple indicate the indices of output of backbone.
> > > - **reduction** (int): Reduction factor of projection transform. Default: 2.
> > > - **use_scale** (bool): Whether to scale pairwise_weight by sqrt(1/inter_channels). Default: False.
> > > - **mode** (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian'.
> > > - **temperature** (float): Temperature to adjust attention. Default: 0.05.
> > > - **concat_input** (bool): Whether concat the input and output of convs before classification layer. Default: True
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.

## [SFNet](../../paddleseg/models/sfnet.py)
> CLASS paddleseg.models.SFNet(num_classes, backbone, backbone_indices, enable_auxiliary_loss=False, align_corners=False, pretrained=None)

    The SFNet implementation based on PaddlePaddle.

    The original article refers to
    Li, Xiangtai, et al. "Semantic Flow for Fast and Accurate Scene Parsing"
    (https://arxiv.org/pdf/2002.10120.pdf).

> > Args
> > > - **num_class** (int): The unique number of target classes.
> > > - **backbone** (Paddle.nn.Layer): A backbone network.
> > > - **backbone_indices** (tuple): The values in the tuple indicate the indices of output of backbone.
> > > - **enable_auxiliary_loss** (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
> > > - **align_corners** (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
> > > - **pretrained** (str, optional): The path or url of pretrained model. Default: None.
