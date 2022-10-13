# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import numpy as np
from functools import partial
from typing import Tuple, List, Union
from scipy.ndimage.filters import gaussian_filter

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

from tools.preprocess_utils import GenericPreprocessor, PreprocessorFor2D
from nnunet.utils.utils import no_op, pad_nd_image
from nnunet.transforms import default_2D_augmentation_params, default_3D_augmentation_params


class StaticBasePredictor:
    """
    Static predictor for nnunet.
    """

    def __init__(self):
        self.input_shape_must_be_divisible_by = None
        self.threeD = None
        self.num_classes = None
        self.inference_apply_nonlin = None

        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def predict_3D(self,
                   x: np.ndarray,
                   do_mirroring: bool,
                   mirror_axes: Tuple[int, ...]=(0, 1, 2),
                   use_sliding_window: bool=False,
                   step_size: float=0.5,
                   patch_size: Tuple[int, ...]=None,
                   regions_class_order: Tuple[int, ...]=None,
                   use_gaussian: bool=False,
                   pad_border_mode: str="constant",
                   pad_kwargs: dict=None,
                   verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: Input data. Must be a nd.ndarray of shape (c, x, y, z).
        do_mirroring: If True, use test time data augmentation in the form of mirroring.
        mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three axes. Default: (0, 1, 2).
        use_sliding_window: if True, run sliding window prediction. Heavily recommended! Default: True.
        step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1! Default: 0.5.
        patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations. Default: None.
        regions_class_order: Return region class by given order. Default: None.
        use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        pad_border_mode: The type of padding image border. Default: 'constant'.
        pad_kwargs: Parameters for padding image border. Default: NOne.
        verbose: Whether print log. Default: True.
        mixed_precision: Whether use amp in inference. Default: True.
        """
        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive predictions'

        if verbose:
            print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if self.threeD:
            if use_sliding_window:
                res = self._internal_predict_3D_3Dconv_tiled(
                    x,
                    step_size,
                    do_mirroring,
                    mirror_axes,
                    patch_size,
                    regions_class_order,
                    use_gaussian,
                    pad_border_mode,
                    pad_kwargs=pad_kwargs,
                    verbose=verbose)
            else:
                res = self._internal_predict_3D_3Dconv(
                    x,
                    patch_size,
                    do_mirroring,
                    mirror_axes,
                    regions_class_order,
                    pad_border_mode,
                    pad_kwargs=pad_kwargs,
                    verbose=verbose)
        else:
            if use_sliding_window:
                res = self._internal_predict_3D_2Dconv_tiled(
                    x, patch_size, do_mirroring, mirror_axes, step_size,
                    regions_class_order, use_gaussian, pad_border_mode,
                    pad_kwargs, False)
            else:
                res = self._internal_predict_3D_2Dconv(
                    x, patch_size, do_mirroring, mirror_axes,
                    regions_class_order, pad_border_mode, pad_kwargs, False)
        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(
            tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(
            gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                          image_size: Tuple[int, ...],
                                          step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)
                ], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        target_step_sizes_in_voxels = [i * step_size for i in patch_size]
        num_steps = [
            int(np.ceil((i - k) / j)) + 1
            for i, j, k in zip(image_size, target_step_sizes_in_voxels,
                               patch_size)
        ]

        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999

            steps_here = [
                int(np.round(actual_step_size * i))
                for i in range(num_steps[dim])
            ]
            steps.append(steps_here)
        return steps

    def _internal_predict_3D_3Dconv_tiled(
            self,
            x: np.ndarray,
            step_size: float,
            do_mirroring: bool,
            mirror_axes: tuple,
            patch_size: tuple,
            regions_class_order: tuple,
            use_gaussian: bool,
            pad_border_mode: str,
            pad_kwargs: dict,
            verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs,
                                    True, None)
        data_shape = data.shape

        steps = self._compute_steps_for_sliding_window(
            patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("step_size:", step_size)
            print("do mirror:", do_mirroring)
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all([
                    i == j
                    for i, j in zip(patch_size,
                                    self._patch_size_for_gaussian_3d)
            ]):
                gaussian_importance_map = self._get_gaussian(
                    patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose:
                    print("computing Gaussian done")
            else:
                if verbose:
                    print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

        else:
            gaussian_importance_map = None

        if use_gaussian and num_tiles > 1:
            add_for_nb_of_preds = self._gaussian_3d
        else:
            add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
        aggregated_results = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                        mirror_axes, do_mirroring, gaussian_importance_map)[0]

                    predicted_patch = predicted_patch
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:
                                       ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:
                                                 ub_z] += add_for_nb_of_preds

        slicer = tuple([
            slice(0, aggregated_results.shape[i])
            for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
        ] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        aggregated_results /= aggregated_nb_of_predictions

        del aggregated_nb_of_predictions
        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(
                class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if verbose:
            print("prediction done")
        return predicted_segmentation, aggregated_results

    def _internal_predict_2D_2Dconv(
            self,
            x: np.ndarray,
            min_size: Tuple[int, int],
            do_mirroring: bool,
            mirror_axes: tuple=(0, 1, 2),
            regions_class_order: tuple=None,
            pad_border_mode: str="constant",
            pad_kwargs: dict=None,
            verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 3, "x must be (c, x, y), but got {}.".format(
            x.shape)

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set but got None'
        if verbose:
            print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs,
                                    True, self.input_shape_must_be_divisible_by)
        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(
            data[None], mirror_axes, do_mirroring, None)[0]

        slicer = tuple([
            slice(0, predicted_probabilities.shape[i])
            for i in range(
                len(predicted_probabilities.shape) - (len(slicer) - 1))
        ] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation
            predicted_probabilities = predicted_probabilities
        else:
            predicted_probabilities = predicted_probabilities
            predicted_segmentation = np.zeros(
                predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_3Dconv(
            self,
            x: np.ndarray,
            min_size: Tuple[int, ...],
            do_mirroring: bool,
            mirror_axes: tuple=(0, 1, 2),
            regions_class_order: tuple=None,
            pad_border_mode: str="constant",
            pad_kwargs: dict=None,
            verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z), but got {}.".format(
            x.shape)

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set but got None'
        if verbose:
            print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs,
                                    True, self.input_shape_must_be_divisible_by)
        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(
            data[None], mirror_axes, do_mirroring, None)[0]

        slicer = tuple([
            slice(0, predicted_probabilities.shape[i])
            for i in range(
                len(predicted_probabilities.shape) - (len(slicer) - 1))
        ] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation
            predicted_probabilities = predicted_probabilities
        else:
            predicted_probabilities = predicted_probabilities
            predicted_segmentation = np.zeros(
                predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D(self,
                                           x: np.ndarray,
                                           mirror_axes: tuple,
                                           do_mirroring: bool=True,
                                           mult: np.ndarray=None):
        assert len(
            x.shape) == 5, 'x must be (b, c, x, y, z), but got {}.'.format(
                x.shape)

        result = np.zeros(
            [1, self.num_classes] + list(x.shape[2:]), dtype='float32')

        if do_mirroring:
            mirror_idx = 8
            num_results = 2**len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (4, ))))
                result += 1 / num_results * np.flip(pred, (4, ))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (3, ))))
                result += 1 / num_results * np.flip(pred, (3, ))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (4, 3))))
                result += 1 / num_results * np.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (2, ))))
                result += 1 / num_results * np.flip(pred, (2, ))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (4, 2))))
                result += 1 / num_results * np.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (3, 2))))
                result += 1 / num_results * np.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (
                    2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (4, 3, 2))))
                result += 1 / num_results * np.flip(pred, (4, 3, 2))

        if mult is not None:
            result[:, :] *= mult
        return result

    def _internal_maybe_mirror_and_pred_2D(self,
                                           x: np.ndarray,
                                           mirror_axes: tuple,
                                           do_mirroring: bool=True,
                                           mult: np.ndarray=None):
        assert len(x.shape) == 4, 'x must be (b, c, x, y), but got {}.'.format(
            x.shape)

        result = np.zeros(
            [x.shape[0], self.num_classes] + list(x.shape[2:]), dtype='float32')

        if do_mirroring:
            mirror_idx = 4
            num_results = 2**len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result += 1 / num_results * pred
            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (3, ))))
                result += 1 / num_results * np.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (2, ))))
                result += 1 / num_results * np.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(np.flip(x, (3, 2))))
                result += 1 / num_results * np.flip(pred, (3, 2))
        if mult is not None:
            result[:, :] *= mult
        return result

    def _internal_predict_2D_2Dconv_tiled(
            self,
            x: np.ndarray,
            step_size: float,
            do_mirroring: bool,
            mirror_axes: tuple,
            patch_size: tuple,
            regions_class_order: tuple,
            use_gaussian: bool,
            pad_border_mode: str,
            pad_kwargs: dict,
            verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 3, "x must be (c, x, y), but got {}.".format(
            x.shape)
        assert patch_size is not None, "patch_size cannot be None for tiled prediction."

        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs,
                                    True, None)
        data_shape = data.shape

        steps = self._compute_steps_for_sliding_window(
            patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("step_size:", step_size)
            print("do mirror:", do_mirroring)
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all([
                    i == j
                    for i, j in zip(patch_size,
                                    self._patch_size_for_gaussian_2d)
            ]):
                if verbose:
                    print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(
                    patch_size, sigma_scale=1. / 8)
                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose:
                    print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

        else:
            gaussian_importance_map = None

        if use_gaussian and num_tiles > 1:
            add_for_nb_of_preds = self._gaussian_2d
        else:
            add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
        aggregated_results = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes,
                    do_mirroring, gaussian_importance_map)[0]

                predicted_patch = predicted_patch
                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:
                                             ub_y] += add_for_nb_of_preds

        slicer = tuple([
            slice(0, aggregated_results.shape[i])
            for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
        ] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(
                class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if verbose:
            print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv(
            self,
            x: np.ndarray,
            min_size: Tuple[int, int],
            do_mirroring: bool,
            mirror_axes: tuple=(0, 1),
            regions_class_order: tuple=None,
            pad_border_mode: str="constant",
            pad_kwargs: dict=None,
            verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z), but got {}.".format(
            x.shape)
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(
                x[:, s], min_size, do_mirroring, mirror_axes,
                regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv_tiled(
            self,
            x: np.ndarray,
            patch_size: Tuple[int, int],
            do_mirroring: bool,
            mirror_axes: tuple=(0, 1),
            step_size: float=0.5,
            regions_class_order: tuple=None,
            use_gaussian: bool=False,
            pad_border_mode: str="edge",
            pad_kwargs: dict=None,
            verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z), but got {}.".format(
            x.shape)
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size,
                regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
                verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred


class StaticPredictor(StaticBasePredictor):
    def __init__(self,
                 model_path,
                 param_path,
                 plans,
                 stage,
                 min_subgraph_size=3):
        super().__init__()
        self.stage = stage
        self.min_subgraph_size = min_subgraph_size
        self.plans = plans
        self.inference_apply_nonlin = lambda x: x
        self.model_path = model_path
        self.param_path = param_path

        self.patch_size = np.array(self.plans['plans_per_stage'][self.stage][
            'patch_size']).astype(int)
        self.input_shape_must_be_divisible_by = np.prod(
            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'],
            0,
            dtype=np.int64)
        self.num_classes = self.plans['num_classes'] + 1

        if len(self.patch_size) == 2:
            self.threeD = False
            self.data_aug_params = default_2D_augmentation_params
        elif len(self.patch_size) == 3:
            self.threeD = True
            self.data_aug_params = default_3D_augmentation_params

        self._init_base_config()
        self._init_gpu_config()
        self.predictor = create_predictor(self.pred_cfg)
        input_names = self.predictor.get_input_names()
        self.input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(output_names[0])

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.model_path, self.param_path)
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_gpu_config(self):
        print("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)

    def __call__(self, x):
        self.input_handle.reshape(x.shape)
        self.input_handle.copy_from_cpu(x)
        self.predictor.run()
        results = self.output_handle.copy_to_cpu()
        return results

    def predict_preprocessed_data_return_seg_and_softmax(
            self,
            data: np.ndarray,
            do_mirroring: bool=True,
            mirror_axes: Tuple[int]=None,
            use_sliding_window: bool=True,
            step_size: float=0.5,
            use_gaussian: bool=True,
            pad_border_mode: str='constant',
            pad_kwargs: dict=None,
            verbose: bool=True,
            mixed_precision=False):
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}
        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']
        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"
        res = self.predict_3D(
            x=data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            patch_size=self.patch_size,
            regions_class_order=None,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            verbose=verbose)
        return res
