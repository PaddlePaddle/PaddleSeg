import numpy as np
import nibabel as nib
import os
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

from medicalseg.cvlibs import manager
from medicalseg.inference_helpers import InferenceHelper

from tools.preprocess_utils.geometry import resize_image, resize_segmentation


@manager.INFERENCE_HELPERS.add_component
class NNFormerInferenceHelper(InferenceHelper):
    def load_medical_data(self, filename):
        self.nimg = nib.load(filename)
        data_array = self.nimg.get_data()
        original_spacing = self.nimg.header["pixdim"][1:4]
        return data_array, original_spacing

    def preprocess(self, cfg, imgs_path, batch_size, batch_id):
        data_array, original_spacing = self.load_medical_data(imgs_path[
            batch_id:(batch_id + batch_size)][0])
        self.shape = data_array.shape
        self.target_spacing = [1.52, 1.52, 6.35]
        new_shape = np.round(((np.array(original_spacing) /
                               np.array(self.target_spacing)).astype(float) *
                              np.array(self.shape))).astype(int)
        data_array = resize_image(data_array, new_shape)
        data_array = np.transpose(data_array, [2, 0, 1])
        mean = np.mean(data_array)
        std = np.std(data_array)
        data_array = (data_array - mean) / (std + 1e-10)
        return np.expand_dims(data_array.astype("float32"), [0, 1])

    def postprocess(self, result):

        result = np.argmax(result, axis=1)
        result = np.transpose(result[0], [1, 2, 0])
        result = resize_segmentation(result, self.shape)
        return np.expand_dims(result, [0])
