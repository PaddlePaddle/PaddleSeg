import paddle
import paddle.nn.functional as F
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import BasePredictor


class BRSBasePredictor(BasePredictor):
    def __init__(self, model, opt_functor, optimize_after_n_clicks=1, **kwargs):
        super().__init__(model, **kwargs)
        self.optimize_after_n_clicks = optimize_after_n_clicks
        self.opt_functor = opt_functor

        self.opt_data = None
        self.input_data = None

    def set_input_image(self, image_nd):
        super().set_input_image(image_nd)
        self.opt_data = None
        self.input_data = None

    def _get_clicks_maps_nd(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros([len(clicks_lists), 1] + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros([len(clicks_lists), 1] + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1

                if click.is_positive:
                    pos_clicks_map[list_indx, 0, y1:y2, x1:x2] = True
                else:
                    neg_clicks_map[list_indx, 0, y1:y2, x1:x2] = True

        with paddle.no_grad():
            pos_clicks_map = paddle.to_tensor(pos_clicks_map)
            neg_clicks_map = paddle.to_tensor(neg_clicks_map)
        return pos_clicks_map, neg_clicks_map

    def get_states(self):
        return {'transform_states': self._get_transform_states(), 'opt_data': self.opt_data}

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.opt_data = states['opt_data']


class FeatureBRSPredictor(BRSBasePredictor):
    def __init__(self, model, opt_functor, insertion_mode='after_deeplab', **kwargs):
        super().__init__(model, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None
        self.model = model

        if self.insertion_mode == 'after_deeplab':
            self.num_channels = model.feature_extractor.ch
        elif self.insertion_mode == 'after_c4':
            self.num_channels = model.feature_extractor.aspp_in_channels
        elif self.insertion_mode == 'after_aspp':
            self.num_channels = model.feature_extractor.ch + 32
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])

        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float64)

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = scale.reshape([bs, -1, 1, 1])
            bias = bias.reshape([bs, -1, 1, 1])
            if self.with_flip:
                scale = scale.tile([2, 1, 1, 1])
                bias = bias.tile([2, 1, 1, 1])

            scaled_backbone_features = self.input_data * scale
            scaled_backbone_features = scaled_backbone_features + bias
            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = F.interpolate(x, mode='bilinear', size=self._c1_features.shape[2:], align_corners=True)
                x = paddle.concat((x, self._c1_features), axis=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)

            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = F.interpolate(pred_logits, size=image_nd.shape[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                       **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]
        with paddle.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = paddle.to_tensor(self.opt_data)
                scale, bias, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(scale, bias)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with paddle.no_grad():
            image_nd, prev_mask = self.net.prepare_input(image_nd)
            coord_features = self.net.get_coord_features(image_nd, prev_mask, points)

            if self.net.rgb_conv is not None:
                x = self.net.rgb_conv(paddle.concat((image_nd, coord_features), axis=1))
                additional_features = None
            elif hasattr(self.net, 'maps_transform'):
                x = image_nd
                additional_features = self.net.maps_transform(coord_features)

            if self.insertion_mode == 'after_c4' or self.insertion_mode == 'after_aspp':
                c1, _, c3, c4 = self.net.feature_extractor.backbone(x, additional_features)
                c1 = self.net.feature_extractor.skip_project(c1)

                if self.insertion_mode == 'after_aspp':
                    x = self.net.feature_extractor.aspp(c4)
                    x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=True)
                    x = paddle.concat((x, c1), axis=1)
                    backbone_features = x
                else:
                    backbone_features = c4
                    self._c1_features = c1
            else:
                backbone_features = self.net.feature_extractor(x, additional_features)[0]

            return backbone_features


class HRNetFeatureBRSPredictor(BRSBasePredictor):
    def __init__(self, model, opt_functor, insertion_mode='A', **kwargs):
        super().__init__(model, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None
        # sself.model = model

        if self.insertion_mode == 'A':
            self.num_channels = sum(k * model.feature_extractor.width for k in [1, 2, 4, 8])
        elif self.insertion_mode == 'C':
            self.num_channels = 2 * model.feature_extractor.ocr_width
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])
        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float64)

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = scale.reshape([bs, -1, 1, 1])
            bias = bias.reshape([bs, -1, 1, 1])
            if self.with_flip:
                scale = scale.tile([2, 1, 1, 1])
                bias = bias.tile([2, 1, 1, 1])

            scaled_backbone_features = self.input_data * scale
            scaled_backbone_features = scaled_backbone_features + bias

            if self.insertion_mode == 'A':
                if self.net.feature_extractor.ocr_width > 0:
                    out_aux = self.net.feature_extractor.aux_head(scaled_backbone_features)
                    out_aux.stop_gradient = False
                    feats = self.net.feature_extractor.conv3x3_ocr(scaled_backbone_features)
                    feats.stop_gradient = False
                    context = self.net.feature_extractor.ocr_gather_head(feats, out_aux)
                    context.stop_gradient = False
                    feats = self.net.feature_extractor.ocr_distri_head(feats, context)
                    feats.stop_gradient = False
                else:
                    feats = scaled_backbone_features
                pred_logits = self.net.feature_extractor.cls_head(feats)
                pred_logits.stop_gradient = False
            elif self.insertion_mode == 'C':
                pred_logits = self.net.feature_extractor.cls_head(scaled_backbone_features)
            else:
                raise NotImplementedError

            pred_logits = F.interpolate(pred_logits, size=image_nd.shape[2:], mode='bilinear',
                                        align_corners=True)
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                       **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]

        with paddle.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = paddle.to_tensor(self.opt_data)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with paddle.no_grad():
            image_nd, prev_mask = self.net.prepare_input(image_nd)
            coord_features = self.net.get_coord_features(image_nd, prev_mask, points)

            if self.net.rgb_conv is not None:
                x = self.net.rgb_conv(paddle.concat((image_nd, coord_features), axis=1))
                additional_features = None
            else:
                x = image_nd
                additional_features = self.net.maps_transform(coord_features)

            feats = self.net.feature_extractor.compute_hrnet_feats(x, additional_features)

            if self.insertion_mode == 'A':
                backbone_features = feats
            elif self.insertion_mode == 'C':
                out_aux = self.net.feature_extractor.aux_head(feats)
                feats = self.net.feature_extractor.conv3x3_ocr(feats)

                context = self.net.feature_extractor.ocr_gather_head(feats, out_aux)
                backbone_features = self.net.feature_extractor.ocr_distri_head(feats, context)
            else:
                raise NotImplementedError

        return backbone_features


class InputBRSPredictor(BRSBasePredictor):
    def __init__(self, model, opt_functor, optimize_target='rgb', **kwargs):
        super().__init__(model, opt_functor=opt_functor, **kwargs)
        self.optimize_target = optimize_target

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_lists, image_nd.shape[2:])
        num_clicks = len(clicks_lists[0])

        if self.opt_data is None or is_image_changed:
            if self.optimize_target == 'dmaps':
                opt_channels = self.net.coord_feature_ch - 1 if self.net.with_prev_mask else self.net.coord_feature_ch
            else:
                opt_channels = 3

            bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]
            self.opt_data = paddle.zeros((bs, opt_channels, image_nd.shape[2], image_nd.shape[3]), dtype='float64')

        def get_prediction_logits(opt_bias):
            input_image, prev_mask = self.net.prepare_input(image_nd)
            dmaps = self.net.get_coord_features(input_image, prev_mask, points_nd)

            if self.optimize_target == 'rgb':
                input_image = input_image + opt_bias

            elif self.optimize_target == 'dmaps':
                if self.net.with_prev_mask:
                    dmaps[:, 1:, :, :] = dmaps[:, 1:, :, :] + opt_bias
                else:
                    dmaps = dmaps + opt_bias

            if self.net.rgb_conv is not None:
                x = self.net.rgb_conv(paddle.concat((input_image, dmaps), axis=1))
                if self.optimize_target == 'all':
                    x = x + opt_bias
                coord_features = None
            elif hasattr(self.net, 'maps_transform'):
                x = input_image
                coord_features = self.net.maps_transform(dmaps)

            pred_logits = self.net.backbone_forward(x, coord_features=coord_features)['instances']
            pred_logits = F.interpolate(pred_logits, size=image_nd.shape[2:], mode='bilinear', align_corners=True)

            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, shape=self.opt_data.shape)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data.cpu().numpy().ravel(),
                                       **self.opt_functor.optimizer_params)

            self.opt_data = paddle.to_tensor(opt_result[0])
            self.opt_data = paddle.reshape(self.opt_data, shape=self.opt_data.shape)

        with paddle.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_vars, _ = self.opt_functor.unpack_opt_params(self.opt_data)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits
