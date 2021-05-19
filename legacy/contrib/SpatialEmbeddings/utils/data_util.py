from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from PIL import Image as PILImage


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


class Cluster:
    def __init__(self, ):
        xm = np.repeat(
            np.linspace(0, 2, 2048)[np.newaxis, np.newaxis, :], 1024, axis=1)
        ym = np.repeat(
            np.linspace(0, 1, 1024)[np.newaxis, :, np.newaxis], 2048, axis=2)
        self.xym = np.vstack((xm, ym))

    def cluster(self, prediction, n_sigma=1, min_pixel=160, threshold=0.5):

        height, width = prediction.shape[1:3]
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = np.tanh(prediction[0:2]) + xym_s
        sigma = prediction[2:2 + n_sigma]
        seed_map = sigmoid_np(prediction[2 + n_sigma:2 + n_sigma + 1])

        instance_map = np.zeros((height, width), np.float32)
        instances = []
        count = 1
        mask = seed_map > 0.5

        if mask.sum() > min_pixel:
            spatial_emb_masked = spatial_emb[np.repeat(mask, \
                                spatial_emb.shape[0], 0)].reshape(2, -1)
            sigma_masked = sigma[np.repeat(mask, n_sigma, 0)].reshape(
                n_sigma, -1)
            seed_map_masked = seed_map[mask].reshape(1, -1)

            unclustered = np.ones(mask.sum(), np.float32)
            instance_map_masked = np.zeros(mask.sum(), np.float32)

            while (unclustered.sum() > min_pixel):

                seed = (seed_map_masked * unclustered).argmax().item()
                seed_score = (seed_map_masked * unclustered).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = np.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = np.exp(-1 * np.sum(
                    (spatial_emb_masked - center)**2 * s, 0))
                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > min_pixel:
                    if unclustered[proposal].sum() / proposal.sum() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = np.zeros((height, width), np.float32)
                        instance_mask[mask.squeeze()] = proposal
                        instances.append(
                            {'mask': (instance_mask.squeeze()*255).astype(np.uint8), \
                            'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        return instance_map, instances


def pad_img(img, dst_shape, mode='constant'):
    img_h, img_w = img.shape[:2]
    dst_h, dst_w = dst_shape
    pad_shape = ((0, max(0, dst_h - img_h)), (0, max(0, dst_w - img_w)))
    return np.pad(img, pad_shape, mode)


def save_for_eval(predictions, infer_shape, im_shape, vis_dir, im_name):
    txt_file = os.path.join(vis_dir, im_name + '.txt')
    with open(txt_file, 'w') as f:
        for id, pred in enumerate(predictions):
            save_name = im_name + '_{:02d}.png'.format(id)
            pred_mask = pad_img(pred['mask'], infer_shape)
            pred_mask = pred_mask[:im_shape[0], :im_shape[1]]
            im = PILImage.fromarray(pred_mask)
            im.save(os.path.join(vis_dir, save_name))
            cl = 26
            score = pred['score']
            f.writelines("{} {} {:.02f}\n".format(save_name, cl, score))
