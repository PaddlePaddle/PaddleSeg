# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddleseg.cvlibs.manager import ComponentManager, BACKBONES, TRANSFORMS, OPTIMIZERS, LOSSES

# NOTE: Models, datasets, and postprocessors are very different in the 
# panoptic segmentation task, compared with the semantic segmentation task,
# while most of the backbones and transforms can be reused.
MODELS = ComponentManager("models")
DATASETS = ComponentManager("datasets")
POSTPROCESSORS = ComponentManager("postprocessors")
RUNNERS = ComponentManager("runners")
