import paddle.nn as nn
from .is_model import ISModel
from .modeling.hrnet_ocr import HighResolutionNet


class HRNetModel(ISModel):
    def __init__(self, width=48, ocr_width=256, small=False, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2D, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}
