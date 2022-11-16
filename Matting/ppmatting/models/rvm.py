# TODO add gpl license


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddleseg
from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()


@manager.MODELS.add_component
class RVM(nn.Layer):
    """
    TODO add annotation
    """
    def __init__(self,
                backbone,
                pretrained=None,
                ):
        super().__init__()

        self.backbone = backbone


    def forward(self, src, r1=None, r2=None, r3=None, r4=None, downsample_ratio=1., segmentation_pass=False):
        feas_backbone = self.backbone(src)
        print(len(feas_backbone))
        for fea in feas_backbone:
            print(fea.shape)
            


if __name__ == '__main__':
    import ppmatting
    backbone = ppmatting.models.MobileNetV3_large_x1_0_os16(
        pretrained='mobilenetv3_large_x1_0_ssld/model.pdparams', 
        out_index=[0, 2, 4],
        return_last_conv=True)
    model = RVM(backbone=backbone)
    print(model)

    img = paddle.rand(shape=(1, 3, 512, 512))
    model(img)
