import paddle
from paddleseg.models import MscaleOCR
from paddleseg.datasets import Cityscapes
import paddleseg.transforms as T
from paddleseg.core.val import evaluate

def main():

    model = MscaleOCR(num_classes=19)
    dic = paddle.load('output/model.pdparams')
    model.set_state_dict(dic)
    val_transforms = [
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]

    # 构建验证集
    val_dataset = Cityscapes(
        dataset_root='data/cityscapes',
        transforms=val_transforms,
        mode='val'
    )


    print(evaluate(model,val_dataset))

if __name__ == '__main__':
    main()