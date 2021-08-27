from paddleseg.utils import get_sys_env, logger
from paddleseg.core import train
import paddle
from model.model_stages_paddle import BiSeNet as STDCSeg
import paddleseg.transforms as T
from paddleseg.datasets import Cityscapes
from paddleseg.models.losses import OhemCrossEntropyLoss
from loss.detail_loss_paddle import DetailAggregateLoss
from utils.train import train
from scheduler.warmup_poly_paddle import Warmup_PolyDecay

def main():

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    # 参数设定
    backbone = 'STDCNet1446'  # STDC2: STDCNet1446 ; STDC1: STDCNet813
    n_classes = 19  # 数据类别数目
    pretrain_path = 'pretrained/STDCNet1446_76.47.pdiparams'  # backbone预训练模型参数
    use_boundary_16 = False
    use_boundary_8 = True  # 论文只用了use_boundary_8，效果最好
    use_boundary_4 = False
    use_boundary_2 = False
    use_conv_last = False

    # 模型导入
    model = STDCSeg(backbone=backbone, n_classes=n_classes, pretrain_model=pretrain_path,
                    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
                    use_boundary_16=use_boundary_16, use_conv_last=use_conv_last)

    # 加载模型参数训练（如果没有预训练参数就把下面两行注释掉）
    # params_state = paddle.load(path='output/best_model/model.pdparams')
    # model.set_dict(params_state)

    # 构建训练用的transforms
    transforms = [
        T.ResizeStepScaling(min_scale_factor=0.125, max_scale_factor=1.5, scale_step_size=0.125),
        T.RandomHorizontalFlip(),
        T.RandomPaddingCrop(crop_size=[1024, 512]),  # Seg50:imgsize=(512,1024); Seg75:imgsize=(768,1536)
        T.RandomDistort(brightness_range=0.5, contrast_range=0.5, saturation_range=0.5),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

    # 构建训练集
    train_dataset = Cityscapes(
        dataset_root='data/cityscapes',  # 修改数据集路径
        transforms=transforms,
        mode='train'
    )

    # 构建验证用的transforms
    transforms_val = [
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

    # 构建验证集
    val_dataset = Cityscapes(
        dataset_root='data/cityscapes',
        transforms=transforms_val,
        mode='val'
    )

    # 设置学习率
    base_lr = 0.01
    # lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=60000,end_lr=0.00001)
    lr = Warmup_PolyDecay(lr_rate=base_lr, warmup_steps=1000, iters=80000, end_lr=1e-5)
    optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=5.0e-4)

    losses = {}
    losses['types'] = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), DetailAggregateLoss()]
    losses['coef'] = [1] * 4

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_scales = 0.5, # miou50
        aug_eval = True,
        optimizer=optimizer,
        # resume_model='output/iter_36400',
        save_dir='output',
        iters=80000,
        batch_size=12,# for single GPU: 4*12
        save_interval=200,
        log_iters=10,
        num_workers=8, # 多线程
        losses=losses,
        use_vdl=True)

if __name__ == '__main__':
    main()
