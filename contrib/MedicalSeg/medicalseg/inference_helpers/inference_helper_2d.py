import numpy as np

from medicalseg.cvlibs import manager
from medicalseg.inference_helpers import InferenceHelper


@manager.INFERENCE_HELPERS.add_component
class InferenceHelper2D(InferenceHelper):
    def preprocess(self, cfg, imgs_path, batch_size, batch_id):
        for img in imgs_path[batch_id:batch_id + batch_size]:
            im_list = []
            imgs = np.load(img)
            imgs = imgs[:, np.newaxis, :, :]
            for i in range(imgs.shape[0]):
                im = imgs[i]
                im = cfg.transforms(im)[0]
                im_list.append(im)
            img = np.concatenate(im_list)
        return img

    def postprocess(self, results):
        results = np.argmax(results, axis=1)
        results = results[np.newaxis, :, :, :, :]
        return results
