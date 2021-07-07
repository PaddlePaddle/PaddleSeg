import numpy as np
from copy import deepcopy
from util.misc import get_labels_with_sizes
from data.transforms import remove_image_only_transforms
from albumentations import ReplayCompose


class DSample:
    def __init__(self, image, encoded_masks, objects=None,
                 objects_ids=None, ignore_ids=None, sample_id=None):
        self.image = image
        self.sample_id = sample_id

        if len(encoded_masks.shape) == 2:
            encoded_masks = encoded_masks[:, :, np.newaxis]
        self._encoded_masks = encoded_masks
        self._ignored_regions = []

        if objects_ids is not None:
            if not objects_ids or not isinstance(objects_ids[0], tuple):
                assert encoded_masks.shape[2] == 1
                objects_ids = [(0, obj_id) for obj_id in objects_ids]

            self._objects = dict()
            for indx, obj_mapping in enumerate(objects_ids):
                self._objects[indx] = {
                    'parent': None,
                    'mapping': obj_mapping,
                    'children': []
                }

            if ignore_ids:
                if isinstance(ignore_ids[0], tuple):
                    self._ignored_regions = ignore_ids
                else:
                    self._ignored_regions = [(0, region_id) for region_id in ignore_ids]
        else:
            self._objects = deepcopy(objects)

        self._augmented = False
        self._soft_mask_aug = None
        self._original_data = self.image, self._encoded_masks, deepcopy(self._objects)

    def augment(self, augmentator):
        self.reset_augmentation()
        aug_output = augmentator(image=self.image, mask=self._encoded_masks)
        self.image = aug_output['image']
        self._encoded_masks = aug_output['mask']

        aug_replay = aug_output.get('replay', None)
        if aug_replay:
            assert len(self._ignored_regions) == 0
            mask_replay = remove_image_only_transforms(aug_replay)
            self._soft_mask_aug = ReplayCompose._restore_for_replay(mask_replay)

        self._compute_objects_areas()
        self.remove_small_objects(min_area=1)

        self._augmented = True

    def reset_augmentation(self):
        if not self._augmented:
            return
        orig_image, orig_masks, orig_objects = self._original_data
        self.image = orig_image
        self._encoded_masks = orig_masks
        self._objects = deepcopy(orig_objects)
        self._augmented = False
        self._soft_mask_aug = None

    def remove_small_objects(self, min_area):
        if self._objects and not 'area' in list(self._objects.values())[0]:
            self._compute_objects_areas()

        for obj_id, obj_info in list(self._objects.items()):
            if obj_info['area'] < min_area:
                self._remove_object(obj_id)

    def get_object_mask(self, obj_id):
        layer_indx, mask_id = self._objects[obj_id]['mapping']
        obj_mask = (self._encoded_masks[:, :, layer_indx] == mask_id).astype(np.int32)
        if self._ignored_regions:
            for layer_indx, mask_id in self._ignored_regions:
                ignore_mask = self._encoded_masks[:, :, layer_indx] == mask_id
                obj_mask[ignore_mask] = -1

        return obj_mask

    def get_soft_object_mask(self, obj_id):
        assert self._soft_mask_aug is not None
        original_encoded_masks = self._original_data[1]
        layer_indx, mask_id = self._objects[obj_id]['mapping']
        obj_mask = (original_encoded_masks[:, :, layer_indx] == mask_id).astype(np.float32)
        obj_mask = self._soft_mask_aug(image=obj_mask, mask=original_encoded_masks)['image']
        return np.clip(obj_mask, 0, 1)

    def get_background_mask(self):
        return np.max(self._encoded_masks, axis=2) == 0

    @property
    def objects_ids(self):
        return list(self._objects.keys())

    @property
    def gt_mask(self):
        assert len(self._objects) == 1
        return self.get_object_mask(self.objects_ids[0])

    @property
    def root_objects(self):
        return [obj_id for obj_id, obj_info in self._objects.items() if obj_info['parent'] is None]

    def _compute_objects_areas(self):
        inverse_index = {node['mapping']: node_id for node_id, node in self._objects.items()}
        ignored_regions_keys = set(self._ignored_regions)

        for layer_indx in range(self._encoded_masks.shape[2]):
            objects_ids, objects_areas = get_labels_with_sizes(self._encoded_masks[:, :, layer_indx])
            for obj_id, obj_area in zip(objects_ids, objects_areas):
                inv_key = (layer_indx, obj_id)
                if inv_key in ignored_regions_keys:
                    continue
                try:
                    self._objects[inverse_index[inv_key]]['area'] = obj_area
                    del inverse_index[inv_key]
                except KeyError:
                    layer = self._encoded_masks[:, :, layer_indx]
                    layer[layer == obj_id] = 0
                    self._encoded_masks[:, :, layer_indx] = layer

        for obj_id in inverse_index.values():
            self._objects[obj_id]['area'] = 0

    def _remove_object(self, obj_id):
        obj_info = self._objects[obj_id]
        obj_parent = obj_info['parent']
        for child_id in obj_info['children']:
            self._objects[child_id]['parent'] = obj_parent

        if obj_parent is not None:
            parent_children = self._objects[obj_parent]['children']
            parent_children = [x for x in parent_children if x != obj_id]
            self._objects[obj_parent]['children'] = parent_children + obj_info['children']

        del self._objects[obj_id]

    def __len__(self):
        return len(self._objects)
