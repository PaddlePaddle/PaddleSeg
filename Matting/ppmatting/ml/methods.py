# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import pymatting
from paddleseg.cvlibs import manager


class BaseMLMatting(object):
    def __init__(self, alpha_estimator, **kargs):
        self.alpha_estimator = alpha_estimator
        self.kargs = kargs

    def __call__(self, image, trimap):
        image = self.__to_float64(image)
        trimap = self.__to_float64(trimap)
        alpha_matte = self.alpha_estimator(image, trimap, **self.kargs)
        return alpha_matte

    def __to_float64(self, x):
        x_dtype = x.dtype
        assert x_dtype in ["float32", "float64"]
        x = x.astype("float64")
        return x


@manager.MODELS.add_component
class CloseFormMatting(BaseMLMatting):
    def __init__(self, **kargs):
        cf_alpha_estimator = pymatting.estimate_alpha_cf
        super().__init__(cf_alpha_estimator, **kargs)


@manager.MODELS.add_component
class KNNMatting(BaseMLMatting):
    def __init__(self, **kargs):
        knn_alpha_estimator = pymatting.estimate_alpha_knn
        super().__init__(knn_alpha_estimator, **kargs)


@manager.MODELS.add_component
class LearningBasedMatting(BaseMLMatting):
    def __init__(self, **kargs):
        lbdm_alpha_estimator = pymatting.estimate_alpha_lbdm
        super().__init__(lbdm_alpha_estimator, **kargs)


@manager.MODELS.add_component
class FastMatting(BaseMLMatting):
    def __init__(self, **kargs):
        lkm_alpha_estimator = pymatting.estimate_alpha_lkm
        super().__init__(lkm_alpha_estimator, **kargs)


@manager.MODELS.add_component
class RandomWalksMatting(BaseMLMatting):
    def __init__(self, **kargs):
        rw_alpha_estimator = pymatting.estimate_alpha_rw
        super().__init__(rw_alpha_estimator, **kargs)


if __name__ == "__main__":
    from pymatting.util.util import load_image, save_image, stack_images
    from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
    import cv2

    root = "/mnt/liuyi22/PaddlePaddle/PaddleSeg/Matting/data/examples/"
    image_path = root + "lemur.png"
    trimap_path = root + "lemur_trimap.png"
    cutout_path = root + "lemur_cutout.png"
    image = cv2.cvtColor(
        cv2.imread(image_path).astype("float64"), cv2.COLOR_BGR2RGB) / 255.0

    cv2.imwrite("image.png", (image * 255).astype('uint8'))
    trimap = load_image(trimap_path, "GRAY")
    print(image.shape, trimap.shape)
    print(image.dtype, trimap.dtype)
    cf = CloseFormMatting()
    alpha = cf(image, trimap)

    # alpha = pymatting.estimate_alpha_lkm(image, trimap)

    foreground = estimate_foreground_ml(image, alpha)

    cutout = stack_images(foreground, alpha)

    save_image(cutout_path, cutout)
