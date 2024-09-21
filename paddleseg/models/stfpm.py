import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager
from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101
import paddle.nn.functional as F
from paddleseg.models.backbones.resnet_ms3 import ResNet_MS3
# from contrib.QualityInspector.qinspector.uad.utils.utils import plot_fig
# from skimage import measure, morphology

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
}


@manager.MODELS.add_component
class STFPM(nn.Layer):

    def __init__(self, num_classes, backbone):
        super(STFPM, self).__init__()
        self.student = backbone
        self.teacher = ResNet_MS3(pretrained=True, arch=backbone.model_name)
        self.teacher.eval()

    def forward(self, x):
        stu = self.student(x)
        if self.teacher.training:
            self.teacher.eval()
        with paddle.no_grad():
            tea = self.teacher(x)
        if self.student.training:
            return [[stu, tea]]
        else:
            score_map = 1.
            t_feat = tea
            s_feat = stu
            for j in range(len(t_feat)):
                t_feat[j] = F.normalize(t_feat[j], axis=1)
                s_feat[j] = F.normalize(s_feat[j], axis=1)
                sm = paddle.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
                sm = F.interpolate(sm,
                                   size=(x.shape[2], x.shape[3]),
                                   mode='bilinear',
                                   align_corners=False)
                # aggregate score map by element-wise product
                score_map = score_map * sm  # layer map
            return [score_map]
