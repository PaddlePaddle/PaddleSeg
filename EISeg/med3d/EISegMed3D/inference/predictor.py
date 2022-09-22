import os
import sys
import logging

logging.getLogger().setLevel(logging.ERROR)

import numpy as np
from copy import deepcopy

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "./.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.inference import create_predictor, Config

from inference.ops import DistMaps3D, ScaleLayer, BatchImageNormalize3D, SigmoidForPred


class Click:
    def __init__(self, is_positive, coords, indx=None):
        if coords is None or is_positive is None:
            raise ValueError(
                "The coord is {}, is_positive is {} and one of them is None, but none of them should be."
            )
        self.coords = coords
        self.is_positive = is_positive
        self.index = None

    @property
    def coords_and_indx(self):
        return (*self.coords, )

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


class BasePredictor(object):
    def __init__(self,
                 model_path,
                 param_path,
                 net_clicks_limit=None,
                 with_mask=True,
                 norm_radius=2,
                 spatial_scale=1.0,
                 device="gpu",
                 enable_mkldnn=False,
                 **kwargs):

        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.with_prev_mask = with_mask
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        if not paddle.in_dynamic_mode():
            paddle.disable_static()
        self.normalization = BatchImageNormalize3D(
            [0.00040428873, ],
            [0.00059983705, ], )

        self.transforms = [SigmoidForPred()]  # apply sigmoid after pred

        # !! Todo Set the radius and spatial_scale here
        self.dist_maps = DistMaps3D(
            norm_radius=norm_radius,
            spatial_scale=spatial_scale,
            cpu_mode=False,
            use_disks=True)

        # init predictor config
        self.pred_cfg = Config(model_path, param_path)
        self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

        if self.device == "cpu":
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        self.predictor = create_predictor(self.pred_cfg)

    def _init_gpu_config(self):
        logging.info("Use NVIDIA GPU")
        self.pred_cfg.enable_use_gpu(100, 0)

    def _init_cpu_config(self):
        logging.info("Use x86 CPU")
        self.pred_cfg.disable_gpu()
        if self.enable_mkldnn:
            logging.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            # self.pred_cfg.set_mkldnn_cache_capacity(10)  # cannot use on MAC
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(10)

    def set_input_image(self, image):  # (1, 12, 512, 512)
        # image is np array or other format including scalar，tuple，list，and paddle.Tensor
        self.original_image = paddle.to_tensor(image).astype(
            "float32") / 255  # 1, 512, 512, 12]

        for transform in self.transforms:
            transform.reset()

        if len(self.original_image.shape) == 4:
            self.original_image = self.original_image.unsqueeze(
                0)  # (1, 1, 12, 512, 512)

        # 默认 concate 一个全0的mask作为prev mask
        self.prev_prediction = paddle.zeros_like(
            self.original_image[:, :1, :, :, :])

        if not self.with_prev_mask:
            self.prev_edge = paddle.zeros_like(self.original_image[:, :
                                                                   1, :, :, :])

    def get_prediction_noclicker(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()  # one click a time todo：累计多个点

        input_image = self.original_image  # [1, 1, 512, 512, 12]
        if prev_mask is None:
            if not self.with_prev_mask:
                prev_mask = self.prev_edge
            else:
                prev_mask = self.prev_prediction
        input_image = paddle.concat(
            [input_image, prev_mask], axis=1)  # [1, 2, 512, 512, 12]

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list])
        pred_logits = self._get_prediction(image_nd, clicks_lists,
                                           is_image_changed)

        pred_logits = paddle.to_tensor(pred_logits)  # [1, 1, 512, 512, 12]
        for t in reversed(self.transforms):  # conv as the final output
            pred_logits = t.inv_transform(pred_logits)

        self.prev_prediction = pred_logits
        return pred_logits.numpy()[0, 0]

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()  #

        input_image = self.original_image
        if prev_mask is None:
            if not self.with_prev_mask:
                prev_mask = self.prev_edge
            else:
                prev_mask = self.prev_prediction

        input_image = paddle.concat([input_image, prev_mask], axis=1)

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list])
        pred_logits = self._get_prediction(image_nd, clicks_lists,
                                           is_image_changed)

        pred_logits = paddle.to_tensor(pred_logits)

        prediction = F.interpolate(
            pred_logits,
            mode="trilinear",
            align_corners=True,
            size=image_nd.shape[2:],
            data_format="NCDHW")

        for t in reversed(self.transforms):
            if pred_edges is not None:
                edge_prediction = t.inv_transform(edge_prediction)
                self.prev_edge = edge_prediction
            prediction = t.inv_transform(prediction)

        self.prev_prediction = prediction
        return prediction.numpy()[0, 0]

    def prepare_input(self, image):
        prev_mask = image[:, 1:, :, :, :]
        image = image[:, :1, :, :, :]
        image = self.normalization(image)
        return image, prev_mask

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)  # [1, 2, 512, 512, 12]

        if prev_mask is not None:
            coord_features = paddle.concat(
                (prev_mask, coord_features), axis=1)  # [1, 3, 512, 512, 12]

        return coord_features

    def _get_prediction(
            self, image_nd, clicks_lists,
            is_image_changed):  # what is the click? click on the right place?
        input_names = self.predictor.get_input_names()
        self.input_handle_1 = self.predictor.get_input_handle(input_names[0])
        self.input_handle_2 = self.predictor.get_input_handle(input_names[1])
        points_nd = self.get_points_nd(clicks_lists)  # 一个正点，一个负点

        image, prev_mask = self.prepare_input(image_nd)

        coord_features = self.get_coord_features(image, prev_mask, points_nd)
        image = image.numpy().astype("float32")
        coord_features = coord_features.numpy().astype("float32")

        # logging.info("coord_features.shape, image.shape", coord_features.shape, image.shape, prev_mask.shape)
        self.input_handle_1.copy_from_cpu(image)
        self.input_handle_2.copy_from_cpu(coord_features)

        self.predictor.run()

        output_names = self.predictor.get_output_names()

        output_handle = self.predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        return output_data

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        logging.info(
            "check_list",
            clicks_lists, )
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list)
            for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]

        num_max_points = max(num_pos_clicks + num_neg_clicks)

        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [
                click.coords_and_indx for click in clicks_list
                if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)
                                       ) * [(-1, -1, -1, -1)]

            neg_clicks = [
                click.coords_and_indx for click in clicks_list
                if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)
                                       ) * [(-1, -1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return paddle.to_tensor(total_clicks)

    def get_states(self):
        return {
            "transform_states": self._get_transform_states(),
            "prev_prediction": self.prev_prediction,
        }

    def set_states(self, states):
        self._set_transform_states(states["transform_states"])
        self.prev_prediction = states["prev_prediction"]
