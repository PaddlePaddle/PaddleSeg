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

from .default_config import default_2D_augmentation_params, default_3D_augmentation_params
from .transform import *
from .single_threaded_augmenter import SingleThreadedAugmenter
from .multi_threaded_augmenter import MultiThreadedAugmenter


def get_moreDA_augmentation(dataloader_train,
                            dataloader_val,
                            patch_size,
                            params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None,
                            seeds_val=None,
                            order_seg=1,
                            order_data=3,
                            deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None,
                            pin_memory=True,
                            regions=None,
                            use_multi_augmenter: bool=False):
    assert params.get(
        'mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(
            DataChannelSelectionTransform(
                params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(
            SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0, )
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=order_seg,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis")))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5))
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel")))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes))
    tr_transforms.append(
        GammaTransform(
            params.get("gamma_range"),
            True,
            True,
            retain_stats=params.get("gamma_retain_stats"),
            p_per_sample=0.1))

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get(
            "mask_was_used_for_normalization")
        tr_transforms.append(
            MaskTransform(
                mask_was_used_for_normalization,
                mask_idx_in_seg=0,
                set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_channel_to_data") is not None and params.get(
            "move_last_seg_channel_to_data"):
        tr_transforms.append(
            MoveSegAsOneHotToData(
                1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get(
                "cascade_do_cascade_augmentations") is not None and params.get(
                    "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(
                            range(-len(params.get("all_segmentation_labels")),
                                  0)),
                        p_per_sample=params.get(
                            "cascade_random_binary_transform_p"),
                        key="data",
                        strel_size=params.get(
                            "cascade_random_binary_transform_size"),
                        p_per_label=params.get(
                            "cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(
                            range(-len(params.get("all_segmentation_labels")),
                                  0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get(
                            "cascade_remove_conn_comp_max_size_percent_threshold"
                        ),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(
            ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(
                DownsampleSegForDSTransform3(deep_supervision_scales, 'target',
                                             'target', classes))
        else:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales,
                    0,
                    input_key='target',
                    output_key='target'))

    tr_transforms = Compose(tr_transforms)
    if not use_multi_augmenter:
        batchgenerator_train = SingleThreadedAugmenter(dataloader_train,
                                                       tr_transforms)
    else:
        batchgenerator_train = MultiThreadedAugmenter(
            dataloader_train, tr_transforms,
            params.get('num_threads'), params.get('num_cached_per_thread'))

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(
            DataChannelSelectionTransform(
                params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(
            SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_channel_to_data") is not None and params.get(
            "move_last_seg_channel_to_data"):
        val_transforms.append(
            MoveSegAsOneHotToData(
                1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(
            ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(
                DownsampleSegForDSTransform3(deep_supervision_scales, 'target',
                                             'target', classes))
        else:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales,
                    0,
                    input_key='target',
                    output_key='target'))

    val_transforms = Compose(val_transforms)

    if not use_multi_augmenter:
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val,
                                                     val_transforms)
    else:
        batchgenerator_val = MultiThreadedAugmenter(
            dataloader_val, val_transforms, params.get('num_threads') // 2, 1)

    return batchgenerator_train, batchgenerator_val
