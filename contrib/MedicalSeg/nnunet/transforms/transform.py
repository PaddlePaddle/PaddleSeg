# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

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

import abc
import numpy as np
from warnings import warn
from .functional import *
from skimage.morphology import ball, label
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening
from copy import deepcopy


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Please implement __call__ method.")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val)
             for key, val in self.__dict__.items()]) + " )"
        return ret_str


class Compose(AbstractTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class DataChannelSelectionTransform(AbstractTransform):
    def __init__(self, channels, data_key="data"):
        self.data_key = data_key
        self.channels = channels

    def __call__(self, **data_dict):
        data_dict[self.data_key] = data_dict[self.data_key][:, self.channels]
        return data_dict


class SegChannelSelectionTransform(AbstractTransform):
    def __init__(self, channels, keep_discarded_seg=False, label_key="seg"):
        self.label_key = label_key
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn(
                "You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning "
                "data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                discarded_seg_idx = [
                    i for i in range(len(seg[0])) if i not in self.channels
                ]
                data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
            data_dict[self.label_key] = seg[:, self.channels]
        return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class SpatialTransform(AbstractTransform):
    def __init__(self,
                 patch_size,
                 patch_center_dist_from_border=30,
                 do_elastic_deform=True,
                 alpha=(0., 1000.),
                 sigma=(10., 13.),
                 do_rotation=True,
                 angle_x=(0, 2 * np.pi),
                 angle_y=(0, 2 * np.pi),
                 angle_z=(0, 2 * np.pi),
                 do_scale=True,
                 scale=(0.75, 1.25),
                 border_mode_data='nearest',
                 border_cval_data=0,
                 order_data=3,
                 border_mode_seg='constant',
                 border_cval_seg=0,
                 order_seg=0,
                 random_crop=True,
                 data_key="data",
                 label_key="seg",
                 p_el_per_sample=1,
                 p_scale_per_sample=1,
                 p_rot_per_sample=1,
                 independent_scale_for_each_axis=False,
                 p_rot_per_axis: float=1,
                 p_independent_scale_per_axis: int=1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size
        ret_val = augment_spatial(
            data,
            seg,
            patch_size=patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            do_elastic_deform=self.do_elastic_deform,
            alpha=self.alpha,
            sigma=self.sigma,
            do_rotation=self.do_rotation,
            angle_x=self.angle_x,
            angle_y=self.angle_y,
            angle_z=self.angle_z,
            do_scale=self.do_scale,
            scale=self.scale,
            border_mode_data=self.border_mode_data,
            border_cval_data=self.border_cval_data,
            order_data=self.order_data,
            border_mode_seg=self.border_mode_seg,
            border_cval_seg=self.border_cval_seg,
            order_seg=self.order_seg,
            random_crop=self.random_crop,
            p_el_per_sample=self.p_el_per_sample,
            p_scale_per_sample=self.p_scale_per_sample,
            p_rot_per_sample=self.p_rot_per_sample,
            independent_scale_for_each_axis=self.
            independent_scale_for_each_axis,
            p_rot_per_axis=self.p_rot_per_axis,
            p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        return data_dict


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class GaussianNoiseTransform(AbstractTransform):
    def __init__(self,
                 noise_variance=(0, 0.1),
                 p_per_sample=1,
                 p_per_channel: float=1,
                 per_channel: bool=False,
                 data_key="data"):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(
                    data_dict[self.data_key][b], self.noise_variance,
                    self.p_per_channel, self.per_channel)
        return data_dict


class GaussianBlurTransform(AbstractTransform):
    def __init__(self,
                 blur_sigma: Tuple[float, float]=(1, 5),
                 different_sigma_per_channel: bool=True,
                 different_sigma_per_axis: bool=False,
                 p_isotropic: float=0,
                 p_per_channel: float=1,
                 p_per_sample: float=1,
                 data_key: str="data"):
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_blur(
                    data_dict[self.data_key][b],
                    self.blur_sigma,
                    self.different_sigma_per_channel,
                    self.p_per_channel,
                    different_sigma_per_axis=self.different_sigma_per_axis,
                    p_isotropic=self.p_isotropic)
        return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self,
                 multiplier_range=(0.5, 2),
                 per_channel=True,
                 data_key="data",
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_brightness_multiplicative(
                    data_dict[self.data_key][b], self.multiplier_range,
                    self.per_channel)
        return data_dict


class BrightnessTransform(AbstractTransform):
    def __init__(self,
                 mu,
                 sigma,
                 per_channel=True,
                 data_key="data",
                 p_per_sample=1,
                 p_per_channel=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(
                    data[b],
                    self.mu,
                    self.sigma,
                    self.per_channel,
                    p_per_channel=self.p_per_channel)

        data_dict[self.data_key] = data
        return data_dict


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(
            self,
            contrast_range: Union[Tuple[float, float], Callable[[], float]]=(
                0.75, 1.25),
            preserve_range: bool=True,
            per_channel: bool=True,
            data_key: str="data",
            p_per_sample: float=1,
            p_per_channel: float=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_contrast(
                    data_dict[self.data_key][b],
                    contrast_range=self.contrast_range,
                    preserve_range=self.preserve_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel)
        return data_dict


class SimulateLowResolutionTransform(AbstractTransform):
    def __init__(self,
                 zoom_range=(0.5, 1),
                 per_channel=False,
                 p_per_channel=1,
                 channels=None,
                 order_downsample=1,
                 order_upsample=0,
                 data_key="data",
                 p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_linear_downsampling_scipy(
                    data_dict[self.data_key][b],
                    zoom_range=self.zoom_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel,
                    channels=self.channels,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes)
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self,
                 gamma_range=(0.5, 2),
                 invert_image=False,
                 per_channel=False,
                 data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]]=False,
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(
                    data_dict[self.data_key][b],
                    self.gamma_range,
                    self.invert_image,
                    per_channel=self.per_channel,
                    retain_stats=self.retain_stats)
        return data_dict


class MirrorTransform(AbstractTransform):
    def __init__(self,
                 axes=(0, 1, 2),
                 data_key="data",
                 label_key="seg",
                 p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError(
                "MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                ret_val = augment_mirroring(data[b], sample_seg, axes=self.axes)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self,
                 dct_for_where_it_was_used,
                 mask_idx_in_seg=1,
                 set_outside_to=0,
                 data_key="data",
                 seg_key="seg"):
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning(
                "mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist"
            )
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


class RemoveLabelTransform(AbstractTransform):
    def __init__(self,
                 remove_label,
                 replace_with=0,
                 input_key="seg",
                 output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict


class MoveSegAsOneHotToData(AbstractTransform):
    def __init__(self,
                 channel_id,
                 all_seg_labels,
                 key_origin="seg",
                 key_target="data",
                 remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.all_seg_labels = all_seg_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_id = channel_id

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        seg = origin[:, self.channel_id:self.channel_id + 1]
        seg_onehot = np.zeros(
            (seg.shape[0], len(self.all_seg_labels), *seg.shape[2:]),
            dtype=seg.dtype)
        for i, l in enumerate(self.all_seg_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1
        target = np.concatenate((target, seg_onehot), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [
                i for i in range(origin.shape[1]) if i != self.channel_id
            ]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin
        return data_dict


class ApplyRandomBinaryOperatorTransform(AbstractTransform):
    def __init__(self,
                 channel_idx,
                 p_per_sample=0.3,
                 any_of_these=(binary_dilation, binary_erosion, binary_closing,
                               binary_opening),
                 key="data",
                 strel_size=(1, 10),
                 p_per_label=1):
        self.p_per_label = p_per_label
        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

        assert not isinstance(
            channel_idx, tuple
        ), "channel_idx of ApplyRandomBinaryOperatorTransform must be list or int, but got tuple."

        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                ch = deepcopy(self.channel_idx)
                np.random.shuffle(ch)
                for c in ch:
                    if np.random.uniform() < self.p_per_label:
                        operation = np.random.choice(self.any_of_these)
                        selem = ball(np.random.uniform(*self.strel_size))
                        workon = np.copy(data[b, c]).astype(int)
                        res = operation(workon, selem).astype(workon.dtype)
                        data[b, c] = res

                        other_ch = [i for i in ch if i != c]
                        if len(other_ch) > 0:
                            was_added_mask = (res - workon) > 0
                            for oc in other_ch:
                                data[b, oc][was_added_mask] = 0
        data_dict[self.key] = data
        return data_dict


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(
        AbstractTransform):
    def __init__(self,
                 channel_idx,
                 key="data",
                 p_per_sample=0.2,
                 fill_with_other_class_p=0.25,
                 dont_do_if_covers_more_than_X_percent=0.25,
                 p_per_label=1):
        self.p_per_label = p_per_label
        self.dont_do_if_covers_more_than_X_percent = dont_do_if_covers_more_than_X_percent
        self.fill_with_other_class_p = fill_with_other_class_p
        self.p_per_sample = p_per_sample
        self.key = key
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in self.channel_idx:
                    if np.random.uniform() < self.p_per_label:
                        workon = np.copy(data[b, c])
                        num_voxels = np.prod(workon.shape, dtype=np.uint64)
                        lab, num_comp = label(workon, return_num=True)
                        if num_comp > 0:
                            component_ids = []
                            component_sizes = []
                            for i in range(1, num_comp + 1):
                                component_ids.append(i)
                                component_sizes.append(np.sum(lab == i))
                            component_ids = [
                                i
                                for i, j in zip(component_ids, component_sizes)
                                if j < num_voxels *
                                self.dont_do_if_covers_more_than_X_percent
                            ]
                            if len(component_ids) > 0:
                                random_component = np.random.choice(
                                    component_ids)
                                data[b, c][lab == random_component] = 0
                                if np.random.uniform(
                                ) < self.fill_with_other_class_p:
                                    other_ch = [
                                        i for i in self.channel_idx if i != c
                                    ]
                                    if len(other_ch) > 0:
                                        other_class = np.random.choice(other_ch)
                                        data[b, other_class][
                                            lab == random_component] = 1
        data_dict[self.key] = data
        return data_dict


class RenameTransform(AbstractTransform):
    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self,
                 regions: dict,
                 seg_key: str="seg",
                 output_key: str="seg",
                 seg_channel: int=0):
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict


class DownsampleSegForDSTransform3(AbstractTransform):
    def __init__(self,
                 ds_scales=(1, 0.5, 0.25),
                 input_key="seg",
                 output_key="seg",
                 classes=None):
        self.classes = classes
        self.output_key = output_key
        self.input_key = input_key
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform3(
            data_dict[self.input_key][:, 0], self.ds_scales, self.classes)
        return data_dict


class DownsampleSegForDSTransform2(AbstractTransform):
    def __init__(self,
                 ds_scales=(1, 0.5, 0.25),
                 order=0,
                 input_key="seg",
                 output_key="seg",
                 axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform2(
            data_dict[self.input_key], self.ds_scales, self.order, self.axes)
        return data_dict


class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None):
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.astype('float16')
            elif self.cast_to == 'float32':
                tensor = tensor.astype('float32')
            elif self.cast_to == 'int':
                tensor = tensor.astype('int64')
            elif self.cast_to == 'bool':
                tensor = tensor.astype('bool')
            else:
                raise ValueError('Unknown value for cast_to: {}'.format(
                    self.cast_to))
        return tensor

    def __call__(self, **data_dict):
        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(paddle.to_tensor(val))
                elif isinstance(val, (list, tuple)) and all(
                    [isinstance(i, np.ndarray) for i in val]):
                    data_dict[
                        key] = [self.cast(paddle.to_tensor(i)) for i in val]
        else:
            for key in self.keys:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = self.cast(paddle.to_tensor(data_dict[key]))
                elif isinstance(data_dict[key], (list, tuple)) and all(
                    [isinstance(i, np.ndarray) for i in data_dict[key]]):
                    data_dict[key] = [
                        self.cast(paddle.to_tensor(i)) for i in data_dict[key]
                    ]

        return data_dict
