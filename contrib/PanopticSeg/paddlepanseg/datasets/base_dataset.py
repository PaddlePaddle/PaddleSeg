# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import copy
import os

import numpy as np
import paddle

from paddleseg.transforms import Compose

from paddlepanseg.cvlibs import build_info_dict

paddle_version = paddle.__version__[:3]
# paddle version < 2.5.0 and not develop
if paddle_version not in ["2.5", "0.0"]:
    from paddle.fluid.dataloader.collate import default_collate_fn
# paddle version >= 2.5.0 or develop
else:
    from paddle.io.dataloader.collate import default_collate_fn


class PanopticDataset(paddle.io.Dataset):
    NO_COLLATION_KEYS = ('gt_fields', 'trans_info', 'image_id')
    NUM_CLASSES = None
    IMG_CHANNELS = 3
    IGNORE_INDEX = 255
    LABEL_DIVISOR = 1000

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 file_list,
                 label_divisor=None,
                 thing_ids=None,
                 num_classes=None,
                 ignore_index=None,
                 separator=' ',
                 no_collation_keys=None):
        super().__init__()
        mode = mode.lower()
        if mode not in ('train', 'val'):
            raise ValueError(
                "mode should be 'train' or 'val' , but got {}.".format(mode))
        self.mode = mode
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = file_list
        self.num_classes = num_classes if num_classes is not None else self.NUM_CLASSES
        self.ignore_index = ignore_index if ignore_index is not None else self.IGNORE_INDEX
        self.label_divisor = label_divisor if label_divisor is not None else self.LABEL_DIVISOR
        self.thing_ids = thing_ids
        self.sep = separator
        self.no_collation_keys = set(self.NO_COLLATION_KEYS)
        if no_collation_keys is not None:
            self.no_collation_keys |= set(no_collation_keys)
        self.sample_list = self._read_sample_list()

    def _read_sample_list(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.sample_list[idx])
        # Inject thing_ids
        sample['thing_ids'] = copy.copy(self.thing_ids)
        if self.mode == 'val':
            # Do not apply sync transforms on GT
            sample['gt_fields'] = []
        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.sample_list)

    def collate(self, batch):
        res = {}
        sample = batch[0]
        for key in self.no_collation_keys:
            if key not in sample:
                continue
            res[key] = [s.pop(key) for s in batch]
        res.update(default_collate_fn(batch))
        return res

    @classmethod
    def convert_id_for_eval(cls, id_):
        return id_

    @classmethod
    def convert_id_for_train(cls, id_):
        return id_

    @classmethod
    def get_colormap(cls):
        return None


class COCOStylePanopticDataset(PanopticDataset):
    CATEGORY_META_INFO = []
    NO_COLLATION_KEYS = PanopticDataset.NO_COLLATION_KEYS + ('ann', )

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 file_list,
                 json_path,
                 label_divisor=1000,
                 thing_ids=None,
                 num_classes=None,
                 ignore_index=255,
                 separator=' ',
                 no_collation_keys=None):
        self.json_path = json_path
        super().__init__(
            mode=mode,
            dataset_root=dataset_root,
            transforms=transforms,
            file_list=file_list,
            label_divisor=label_divisor,
            thing_ids=thing_ids,
            num_classes=num_classes,
            ignore_index=ignore_index,
            separator=separator,
            no_collation_keys=no_collation_keys)
        if self.num_classes is None:
            self.num_classes = self._get_num_classes()
        if self.thing_ids is None:
            self.thing_ids = self._get_thing_ids()

    def _read_sample_list(self):
        json_path = self.json_path
        file_list = self.file_list
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        ann_dict = {item['image_id']: item for item in json_info['annotations']}
        assert len(ann_dict) == len(json_info['annotations'])
        img_dict = {item['id']: item for item in json_info['images']}
        assert len(img_dict) == len(json_info['images'])

        sample_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.strip().split(self.sep)
                if len(items) != 2:
                    if self.mode == 'train' or self.mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(self.sep))
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                image_id = self._get_image_id(image_path)
                ann_info = ann_dict[image_id]
                img_info = img_dict[image_id]
                sample = build_info_dict(
                    _type_='sample',
                    img=image_path,
                    label=label_path,
                    img_path=image_path,
                    lab_path=label_path)
                sample['img_h'] = img_info['height']
                sample['img_w'] = img_info['width']
                sample['image_id'] = image_id
                seg_info = ann_info['segments_info']
                for item in seg_info:
                    item['category_id'] = self.convert_id_for_train(item[
                        'category_id'])
                seg_info = [
                    item for item in seg_info
                    if item['category_id'] != self.ignore_index
                ]
                sample['ann'] = seg_info
                sample_list.append(sample)
        return sample_list

    @staticmethod
    def _get_image_id(image_path):
        raise NotImplementedError

    @classmethod
    def convert_id_for_eval(cls, id_):
        # Do NOT use `hasattr` here because we expect
        # only the derived class to have this attribute.
        if not '_id_rev_mapper' in cls.__dict__:
            cls._set_id_mapper()
        return cls._id_rev_mapper[id_]

    @classmethod
    def convert_id_for_train(cls, id_):
        if not '_id_mapper' in cls.__dict__:
            cls._set_id_mapper()
        return cls._id_mapper[id_]

    @classmethod
    def _set_id_mapper(cls):
        mapper = {}
        for i, cat in enumerate(cls.CATEGORY_META_INFO):
            mapper[cat['id']] = i
        cls._id_mapper = mapper
        cls._id_rev_mapper = {v: k for k, v in cls._id_mapper.items()}

    @classmethod
    def _get_num_classes(cls):
        return len(cls.CATEGORY_META_INFO)

    @classmethod
    def _get_thing_ids(cls):
        thing_ids = []
        for i, cat in enumerate(cls.CATEGORY_META_INFO):
            if cat['isthing']:
                thing_ids.append(i)
        return thing_ids

    @classmethod
    def get_colormap(cls):
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i, cat in enumerate(cls.CATEGORY_META_INFO):
            colormap[i] = cat['color']
        return colormap
