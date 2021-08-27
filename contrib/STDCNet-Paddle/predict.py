# 参数设定
import paddle
import os
from model.model_stages_paddle import BiSeNet as STDCSeg
import paddleseg.transforms as T
from paddleseg.datasets import Cityscapes
from paddleseg.models.losses import OhemCrossEntropyLoss
from loss.detail_loss_paddle import DetailAggregateLoss
from paddleseg.core import predict
from scheduler.warmup_poly_paddle import Warmup_PolyDecay

def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir

# 参数设定
backbone = 'STDCNet1446' # STDC2: STDCNet1446 ; STDC1: STDCNet813
n_classes = 19 # 数据类别数目
pretrain_path = None # backbone预训练模型参数
use_boundary_16 = False
use_boundary_8 = True # 论文只用use_boundary_8
use_boundary_4 = False
use_boundary_2 = False
use_conv_last = False

# 模型导入
model = STDCSeg(backbone=backbone, n_classes=n_classes, pretrain_model=pretrain_path,
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
    use_boundary_16=use_boundary_16, use_conv_last=use_conv_last)

#加载模型参数训练
model_path='output/best_model/model.pdparams'
image_path = 'images/xxx' #'The path of image, it can be a file or a directory including images'
# 构建transforms
transforms = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
image_list, image_dir = get_image_list(image_path)

if __name__=='__main__':
    predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='predict_output', # output path
            aug_pred=True,
            scales=0.5,
            flip_horizontal=False,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None)