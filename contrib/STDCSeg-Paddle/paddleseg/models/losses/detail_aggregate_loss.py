import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager

def dice_loss_func(input, target):
    smooth = 1.
    n = input.shape[0]
    iflat = paddle.reshape(input, [n, -1])
    tflat = paddle.reshape(target,[n,-1])
    intersection = paddle.sum((iflat * tflat),axis=1)
    loss = 1 - ((2. * intersection + smooth) /
                (paddle.sum(iflat,axis=1) + paddle.sum(tflat,axis=1) + smooth))
    return paddle.mean(loss)

@manager.LOSSES.add_component
class DetailAggregateLoss(nn.Layer):
    """
    Loss for Rethinking BiSeNet (CVPR2021) implementation based on PaddlePaddle.
    paper: Rethinking BiSeNet For Real-time Semantic Segmentation.(https://arxiv.org/abs/2104.13188)
    """
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        self.laplacian_kernel = paddle.to_tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],dtype='float32').reshape((1,1,3,3))
        self.fuse_kernel = paddle.create_parameter([1,3,1,1],dtype='float32')

    def forward(self, boundary_logits, gtmasks):
        """
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        Returns: loss
        """
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