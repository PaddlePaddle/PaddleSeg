import math
import paddle
from paddleseg.models import MscaleOCR
from paddleseg.models import BiSeNetV2
from paddleseg.datasets import Cityscapes
import paddleseg.transforms as T
from paddleseg.models.losses import RMILoss
from paddleseg.models.losses import CrossEntropyLoss
from paddleseg.core import train
import paddle.distributed as dist


def main():
    #dist.init_parallel_env()

    model = MscaleOCR(num_classes=19)
    #model = BiSeNetV2(num_classes=19)
    # 设置支持多卡训练
    #model = paddle.DataParallel(model)

    # 构建训练用的transforms
    train_transforms = [
        T.RandomHorizontalFlip(), T.RandomSizeAndCrop(
            crop_size=[1024, 2048], crop_nopad=False), T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # 构建训练集
    train_dataset = Cityscapes(
        dataset_root='data/cityscapes',
        transforms=train_transforms,
        mode='train')

    val_transforms = [
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # 构建验证集
    val_dataset = Cityscapes(
        dataset_root='data/cityscapes', transforms=val_transforms, mode='val')

    # 设置学习率
    #base_lr = 0.01
    #lr = paddle.optimizer.lr.LambdaDecay(learning_rate= 0.005,lr_lambda= lambda epoch: (1-epoch/175)**2)
    lr = paddle.optimizer.lr.LambdaDecay(
        learning_rate=0.005,
        lr_lambda=lambda iter: (1 - math.ceil(iter / 2975) / 175)**2)
    #lr = paddle.optimizer.lr.LambdaDecay(learning_rate= 0.005,lr_lambda= lambda iter: (1-math.ceil(iter/743)/175)**2)
    optimizer = paddle.optimizer.Momentum(
        lr, parameters=model.parameters(), momentum=0.9, weight_decay=0.0001)

    #设置损失函数
    losses = {}
    #losses['types'] = [
    #RMILoss(num_classes=19,ignore_index=255,do_rmi=False),
    #RMILoss(num_classes=19,ignore_index=255,do_rmi=True),
    #RMILoss(num_classes=19,ignore_index=255,do_rmi=False),
    #RMILoss(num_classes=19,ignore_index=255,do_rmi=False)
    #] 
    losses['types'] = [
        RMILoss(
            num_classes=19, ignore_index=255, do_rmi=False), RMILoss(
                num_classes=19, ignore_index=255, do_rmi=True), RMILoss(
                    num_classes=19, ignore_index=255, do_rmi=False), RMILoss(
                        num_classes=19, ignore_index=255, do_rmi=False)
    ]
    losses['coef'] = [1, 0.4, 0.05, 0.05]

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        resume_model='output/iter_78000',
        save_dir='/root/paddlejob/workspace/output/',
        iters=2975 * 40,
        batch_size=1,
        save_interval=3000,
        log_iters=10,
        num_workers=4,
        losses=losses,
        use_vdl=False)


if __name__ == '__main__':
    main()
