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

from uapi import PaddleModel, Config
from uapi.tests.smoke.seg_testing_template import test_model

if __name__ == '__main__':
    model_name = 'segformer_b0'

    config = Config(model_name)
    # Set dataset params
    config.update({
        'train_dataset': {
            'num_classes': 2,
            'transforms': [{
                'type': 'Resize',
                'target_size': [398, 224]
            }, {
                'type': 'RandomHorizontalFlip'
            }, {
                'type': 'RandomDistort',
                'brightness_range': 0.4,
                'contrast_range': 0.4,
                'saturation_range': 0.4
            }, {
                'type': 'Normalize'
            }],
            'mode': 'train'
        },
        'val_dataset': {
            'num_classes': 2,
            'transforms': [{
                'type': 'Resize',
                'target_size': [398, 224]
            }, {
                'type': 'Normalize'
            }],
            'mode': 'val'
        },
    })
    # Update model params
    config.update({'model': {'num_classes': 2}})

    model = PaddleModel(config=config)

    test_model(model)
