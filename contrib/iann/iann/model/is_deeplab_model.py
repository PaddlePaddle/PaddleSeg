import paddle.nn as nn

from util.serialization import serialize
from .is_model import ISModel
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead
from model.modifiers import LRMult


class DeeplabModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone="resnet50",
        deeplab_ch=256,
        aspp_dropout=0.5,
        backbone_norm_layer=None,
        backbone_lr_mult=0.1,
        norm_layer=nn.BatchNorm2D,
        **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = DeepLabV3Plus(
            backbone=backbone,
            ch=deeplab_ch,
            project_dropout=aspp_dropout,
            norm_layer=norm_layer,
            backbone_norm_layer=backbone_norm_layer,
        )
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))
        self.head = SepConvHead(
            1,
            in_channels=deeplab_ch,
            mid_channels=deeplab_ch // 2,
            num_layers=2,
            norm_layer=norm_layer,
        )

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.feature_extractor(image, coord_features)

        return {"instances": self.head(backbone_features[0])}
