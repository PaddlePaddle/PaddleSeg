from collections import OrderedDict
import paddle
import paddle.nn.functional as F
from paddleseg.utils.ramps import *
from paddleseg.utils.metrics import eval_metrics,get_seg_metrics
from paddleseg.utils import logger


# 预热训练
def warm_up(model, sup_loader, optimizer1, optimizer2, optimizer_s, id):
    model.train()
    total_correct = 0  # 一个epoch预测正确数
    total_label = 0  # 一个epoch,label数
    total_inter = 0  # 一个epoch预测值与标签交集
    total_union = 0  # 一个epoch预测值与标签并集
    loss_epoch=0
    for data in sup_loader:
        (input_l_wk, input_l_str, target_l) = data['weak_aug'], data['strong_aug'], data['label']
        # we only train the strong augmented student
        input_l = input_l_wk if id == 1 or id == 2 else input_l_str

        input_l, target_l = input_l.cuda(blocking=False), target_l.cuda(blocking=False).astype('int64')
        output_l, _ = model(x_l=input_l, x_ul=None, id=id, warm_up=True)
        # 计算监督损失
        loss = F.cross_entropy(output_l, target_l, ignore_index=255, axis=1)
        total_loss = loss.mean()
        loss_epoch = total_loss.item()
        total_loss.backward()
        if id == 1:
            optimizer1.step()
            optimizer1.clear_grad()
        elif id == 2:
            optimizer2.step()
            optimizer2.clear_grad()
        else:
            optimizer_s.step()
            optimizer_s.clear_grad()

        batch_pix_correct, batch_pix_label, batch_inter, batch_union = eval_metrics(output_l, target_l, 21, 255)
        total_correct = total_correct + batch_pix_correct
        total_label = total_label + batch_pix_label
        total_inter = total_inter + batch_inter
        total_union = total_union + batch_union

        del input_l, target_l
        del total_loss
    pixAcc, Iou, mIou = get_seg_metrics(total_correct, total_label, total_inter, total_union)

    infor = "[warm_up] #Images:pixAcc: {:.4f} mIou:{:}  Loss:{:}".format(
        pixAcc[0],mIou,loss_epoch)
    logger.info(infor)
    return


# 不需要计算梯度，也不会进行反向传播
@paddle.no_grad()
def update_teachers(model, teacher_encoder, teacher_decoder, keep_rate=0.996):
    student_encoder_dict = model.encoder_s.state_dict()
    student_decoder_dict = model.decoder_s.state_dict()
    new_teacher_encoder_dict = OrderedDict()
    new_teacher_decoder_dict = OrderedDict()

    for key, value in teacher_encoder.state_dict().items():
        if key in student_encoder_dict.keys():
            new_teacher_encoder_dict[key] = (
                    student_encoder_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student encoder model".format(key))

    for key, value in teacher_decoder.state_dict().items():
        if key in student_decoder_dict.keys():
            new_teacher_decoder_dict[key] = (
                    student_decoder_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student decoder model".format(key))
    teacher_encoder.set_state_dict(new_teacher_encoder_dict, use_structured_name=True)
    teacher_decoder.set_state_dict(new_teacher_decoder_dict, use_structured_name=True)


# 不需要实例化，直接类名.方法名()来调用
def rand_bbox_1(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# 随机混合
def cut_mix(labeled_image, labeled_mask,
            unlabeled_image=None, unlabeled_mask=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()

    # 随机打乱序列
    u_rand_index = paddle.randperm(unlabeled_image.shape[0])[:unlabeled_image.shape[0]].cuda().tolist()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_image.shape, lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask

    return labeled_image, labeled_mask, mix_unlabeled_image, mix_unlabeled_target


def predict_with_out_grad(model, image):
    with paddle.no_grad():
        predict_target_ul1 = model.decoder1(model.encoder1(image),
                                            data_shape=[image.shape[-2], image.shape[-1]])
        predict_target_ul2 = model.decoder2(model.encoder2(image),
                                            data_shape=[image.shape[-2], image.shape[-1]])
        predict_target_ul1 = paddle.nn.functional.interpolate(predict_target_ul1,
                                                              size=(image.shape[-2], image.shape[-1]),
                                                              mode='bilinear',
                                                              align_corners=True)

        predict_target_ul2 = paddle.nn.functional.interpolate(predict_target_ul2,
                                                              size=(image.shape[-2], image.shape[-1]),
                                                              mode='bilinear',
                                                              align_corners=True)

        assert predict_target_ul1.shape == predict_target_ul2.shape, "Expect two prediction in same shape,"
    return predict_target_ul1, predict_target_ul2


# NOTE: the func in here doesn't bring improvements, but stabilize the early stage's training curve.
def assist_mask_calculate(core_predict, assist_predict, topk=1, num_classes=21):
    _, index = paddle.topk(assist_predict, k=topk, axis=1)
    mask = paddle.nn.functional.one_hot(index.squeeze(), num_classes)

    # k!= 1, sum them
    mask = mask.sum(axis=1) if topk > 1 else mask

    if mask.shape[-1] != num_classes:
        mask = paddle.concat((mask, paddle.zeros([mask.shape[0], mask.shape[1],
                                                  mask.shape[2], num_classes - mask.shape[-1]]).cuda()),
                             axis=3)

    mask = paddle.transpose(mask, perm=[0, 3, 1, 2])

    # get the topk result of the assist map
    assist_predict = paddle.multiply(assist_predict, mask)

    # fullfill with core predict value for the other entries;
    # as it will be merged based on threshold value

    # assist_predict[paddle.where(assist_predict == .0)] = core_predict[paddle.where(assist_predict == .0)]
    assist_predict[(assist_predict == .0)] = core_predict[(assist_predict == .0)]
    return assist_predict


