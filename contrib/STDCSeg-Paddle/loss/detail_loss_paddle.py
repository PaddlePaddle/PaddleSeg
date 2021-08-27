import paddle
import paddle.nn as nn
from paddle.nn import functional as F

class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        smooth (float32): laplace smoothing,
            to smooth dice loss and accelerate convergence. following:
            https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
    """

    def __init__(self, ignore_index=255, smooth=0.):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5
        self.smooth = smooth

    def dice_loss_func(self,logits, labels):
        labels = paddle.cast(labels, dtype='int32')
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        labels_one_hot = paddle.transpose(labels_one_hot, [0, 3, 1, 2])
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')

        logits = F.softmax(logits, axis=1)

        mask = (paddle.unsqueeze(labels, 1) != self.ignore_index)
        logits = logits * mask
        labels_one_hot = labels_one_hot * mask

        dims = (0, ) + tuple(range(2, labels.ndimension() + 1))

        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) /
                     (cardinality + self.eps + self.smooth)).mean()
        return 1 - dice_loss

def dice_loss_func(input, target):
    smooth = 1.
    n = input.shape[0]
    iflat = paddle.reshape(input, [n, -1])
    tflat = paddle.reshape(target,[n,-1])
    intersection = paddle.sum((iflat * tflat),axis=1)
    loss = 1 - ((2. * intersection + smooth) /
                (paddle.sum(iflat,axis=1) + paddle.sum(tflat,axis=1) + smooth))
    return paddle.mean(loss)


class DetailAggregateLoss(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        self.laplacian_kernel = paddle.to_tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],dtype='float32').reshape((1,1,3,3))
        self.fuse_kernel = paddle.create_parameter([1,3,1,1],dtype='float32')

    def forward(self, boundary_logits, gtmasks):
        boundary_targets = F.conv2d(paddle.unsqueeze(gtmasks,axis=1).astype('float32'),self.laplacian_kernel, padding=1)
        boundary_targets =paddle.clip(boundary_targets,min=0)
        boundary_targets = boundary_targets>0.1
        boundary_targets = boundary_targets.astype('float32')

        boundary_targets_x2 = F.conv2d(paddle.unsqueeze(gtmasks,axis=1).astype('float32'), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 =paddle.clip(boundary_targets_x2,min=0)
        boundary_targets_x4 = F.conv2d(paddle.unsqueeze(gtmasks,axis=1).astype('float32'), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 =paddle.clip(boundary_targets_x4,min=0)

        boundary_targets_x8 = F.conv2d(paddle.unsqueeze(gtmasks,axis=1).astype('float32'), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 =paddle.clip(boundary_targets_x8,min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up = boundary_targets_x2_up>0.1
        boundary_targets_x2_up = boundary_targets_x2_up.astype('float32')

        boundary_targets_x4_up = boundary_targets_x4_up>0.1
        boundary_targets_x4_up = boundary_targets_x4_up.astype('float32')

        boundary_targets_x8_up = boundary_targets_x8_up>0.1
        boundary_targets_x8_up = boundary_targets_x8_up.astype('float32')

        boudary_targets_pyramids = paddle.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               axis=1)

        boudary_targets_pyramids = paddle.squeeze(boudary_targets_pyramids,axis=2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid = boudary_targets_pyramid>0.1
        boudary_targets_pyramid = boudary_targets_pyramid.astype('float32')

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(F.sigmoid(boundary_logits), boudary_targets_pyramid)
        detail_loss = bce_loss + dice_loss
        gtmasks.stop_gradient = True
        return detail_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params


# if __name__ == '__main__':
#     # torch.manual_seed(15)
#     # with open('../cityscapes_info.json', 'r') as fr:
#     #     labels_info = json.load(fr)
#     # lb_map = {el['id']: el['trainId'] for el in labels_info}
#     #
#     # img_path = 'data/gtFine/val/frankfurt/frankfurt_000001_037705_gtFine_labelIds.png'
#     # img = cv2.imread(img_path, 0)
#     #
#     # label = np.zeros(img.shape, np.uint8)
#     # for k, v in lb_map.items():
#     #     label[img == k] = v
#     #
#     # img_tensor = torch.from_numpy(label).cuda()
#     # img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)
#     #
#     # detailAggregateLoss = DetailAggregateLoss()
#     # for param in detailAggregateLoss.parameters():
#     #     print(param)
#     #
#     # bce_loss, dice_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
#     # print(bce_loss, dice_loss)