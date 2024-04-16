import paddle
import paddle.nn as nn
import numpy as np

import paddle.nn.functional as F


class BaseTransform(object):

    def __init__(self):
        self.image_changed = False

    def transform(self, image_nd, clicks_lists):
        raise NotImplementedError

    def inv_transform(self, prob_map):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class SigmoidForPred(BaseTransform):

    def transform(self, image_nd, clicks_lists):
        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        return F.sigmoid(prob_map)

    def reset(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass


class BatchImageNormalize3D:  # 标准化 均值为0，方差为1

    def __init__(self, mean, std):
        self.mean = paddle.to_tensor(
            np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis,
                           np.newaxis]).astype("float32")
        self.std = paddle.to_tensor(
            np.array(std)[np.newaxis, :, np.newaxis, np.newaxis,
                          np.newaxis]).astype("float32")

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / self.std
        return tensor


class ScaleLayer(nn.Layer):

    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = self.create_parameter(
            shape=[1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(init_value / lr_mult))

    def forward(self, x):
        scale = paddle.abs(self.scale * self.lr_mult)
        return x * scale


class DistMaps3D(nn.Layer):

    def __init__(self,
                 norm_radius,
                 spatial_scale=1.0,
                 cpu_mode=False,
                 use_disks=False):  # (1, 1.0, False, True)
        super(DistMaps3D, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks

        if self.cpu_mode:
            from util.cython import get_dist_maps

            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols,
                           layers):  # [B, num_points*2, 4]
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(
                    self._get_dist_maps(points[i].numpy().astype("float32"),
                                        rows, cols, norm_delimeter))
            coords = paddle.to_tensor(np.stack(coords,
                                               axis=0)).astype("float32")
        else:
            num_points = points.shape[1] // 2  # [1, 2, 4]
            points = points.reshape([-1,
                                     points.shape[2]])  # [B*num_points*2, 4]
            points, points_order = paddle.split(points, [3, 1],
                                                axis=1)  # [2, 3]
            # [B*num_points*2, 3],  [B*num_points*2, 1]
            invalid_points = paddle.max(points, axis=1, keepdim=False) < 0
            row_array = paddle.arange(start=0,
                                      end=rows,
                                      step=1,
                                      dtype="float32")
            col_array = paddle.arange(start=0,
                                      end=cols,
                                      step=1,
                                      dtype="float32")
            layer_array = paddle.arange(start=0,
                                        end=layers,
                                        step=1,
                                        dtype="float32")

            coord_rows, coord_cols, coor_layers = paddle.meshgrid(
                row_array,
                col_array,
                layer_array  # [512, 512, 12]
            )  # len is 3 [rows, cols, layers]
            coords = paddle.unsqueeze(
                paddle.stack([coord_rows, coord_cols, coor_layers], axis=0),
                axis=0).tile(  # [B*num_points*2, 3, rows, cols, layers]
                    [points.shape[0], 1, 1, 1,
                     1])  # [B*num_points*2 | 768, 3, 512, 512, 12] # repeat

            add_xy = (points * self.spatial_scale).reshape(
                [points.shape[0], points.shape[1], 1, 1, 1])
            # [B*num_points*2, 3, 1, 1, 1]
            # 所有的坐标组合，减去point的数值，只有point对应位置为0，其他相近的也小 [B*num_points*2, 3, rows, cols, layers]
            coords = coords - add_xy  # [B*num_points*2, 3, 512, 512, 12]

            if not self.use_disks:
                coords = coords / (self.norm_radius * self.spatial_scale)

            coords = coords * coords  # [B*num_points*2, 3, 512, 512, 12] 取平方
            coords[:, 0] += coords[:, 1] + coords[:, 2]
            coords = coords[:, :1]  # [B*num_points*2, 1, rows, cols, layers]

            # [B*2, num_points, 1, rows, cols, layers]
            coords = coords.reshape([-1, num_points, 1, rows, cols, layers])
            # [B*2, 1, 512, 512, 12] 所有point中最小的
            coords = paddle.min(coords, axis=1)
            coords = coords.reshape([-1, 2, rows, cols, layers])
            #  [B, 2, rows, cols, layers] [B, 2, 512, 512, 12]

        if self.use_disks:
            coords = (coords
                      <= (self.norm_radius * self.spatial_scale)**2).astype(
                          "float32")  # 只取较小的数值对应的特征
        else:
            coords = paddle.tanh(paddle.sqrt(coords) * 2)

        return coords

    def forward(self, x, coords):  #  [16, 1, 512, 512, 12], [16, 48, 4]
        batchsize = x.shape[0]
        rows, cols, layers = x.shape[2:5]
        return self.get_coord_features(coords, batchsize, rows, cols, layers)
