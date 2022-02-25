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

import numpy as np
import SimpleITK as sitk

def load_slices(dcm_dir):
    """
    Load dcm like images
    Return img array and [z,y,x]-ordered origin and spacing
    """

    dcm_list = [os.path.join(dcm_dir, i) for i in os.listdir(dcm_dir)]
    indices = np.array([pydicom.dcmread(i).InstanceNumber for i in dcm_list])
    dcm_list = np.array(dcm_list)[indices.argsort()]

    itkimage = sitk.ReadImage(dcm_list)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def load_series(mhd_path):
    """
    Load mhd, nii like images
    Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(mhd_path)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing