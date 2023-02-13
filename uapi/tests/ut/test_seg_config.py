# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import tempfile

import yaml

from uapi import Config
from uapi.tests.ut.testing_utils import CpuCommonTest


class TestSegConfig(CpuCommonTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # We test `SegConfig` with a registered *real* model
        cls.proto_cfg = Config(model_name='pphumanseg_lite')

    def setUp(self):
        self.cfg = self.proto_cfg.copy()

    def test_update(self):
        # Simple dict
        self.cfg.update({'dict': None})
        # Nested dict
        self.cfg.update({'nested': {'dict': None}})
        # Dict with list values
        self.cfg.update({'list': [1, 2, 3]})
        # List
        with self.assertRaises(AttributeError):
            self.cfg.update([1, 2, 3])

    def test_load(self):
        # We test using customized config file
        with tempfile.TemporaryDirectory() as td:
            dummy_cfg_path = os.path.join(td, 'dummy.yaml')
            dummy_cfg = {
                'train_dataset': {
                    'Type': 'dummy',
                    'transforms': [{
                        'type': 'RandomCrop'
                    }, {
                        'type': 'Resize',
                        'target_size': [256, 256]
                    }]
                }
            }
            with open(dummy_cfg_path, 'w') as f:
                yaml.dump(dummy_cfg, f)
            # Re-use `cfg`
            self.cfg.load(dummy_cfg_path)
            self.assertEqual(dummy_cfg, self.cfg.dict)

    def test_dump(self):
        with tempfile.TemporaryDirectory() as td:
            dummy_cfg_path = os.path.join(td, 'dummy.yaml')
            # We have no way to judge the correctness of the dumped content
            # Therefore, we only test if this does not trigger errors
            self.cfg.dump(dummy_cfg_path)

    def test_update_dataset(self):
        self.cfg.update_dataset('dummy')
        self.assertEqual(self.cfg.train_dataset['type'], 'Dataset')
        self.assertEqual(self.cfg.train_dataset['dataset_root'], 'dummy')
        self.assertEqual(self.cfg.val_dataset['type'], 'Dataset')
        self.assertEqual(self.cfg.train_dataset['dataset_root'], 'dummy')
        with self.assertRaises(ValueError):
            self.cfg.update_dataset('dummy2', dataset_type='Cityscapes')

    def test_update_optimizer(self):
        pass

    def test_update_backbone(self):
        pass

    def test_update_lr_scheduler(self):
        pass

    def test_update_batch_size(self):
        pass

    def test_update_weight_decay(self):
        pass
