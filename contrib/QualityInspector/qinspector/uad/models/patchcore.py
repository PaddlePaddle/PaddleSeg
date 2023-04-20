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

import numpy as np
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import resnet18, resnet50, wide_resnet50_2

from scipy.ndimage import gaussian_filter
from qinspector.cvlib.workspace import register
from qinspector.uad.utils.utils import cdist, cholesky_inverse, mahalanobis, mahalanobis_einsum, orthogonal, svd_orthogonal
from qinspector.uad.utils.k_center_greedy import KCenterGreedy

models = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2,
}
fins = {
    "resnet18": 448,
    "resnet50": 1792,
    "wide_resnet50_2": 1792,
}


def get_projection(fin, fout, method='ortho'):
    if 'sample' == method:
        W = paddle.randperm(fin)[:fout]
        # W = paddle.eye(fin)[W.tolist()].T
    elif 'coreset' == method:
        W = None
    elif 'h_sample' == method:
        s = paddle.randperm(fin // 7)[:fout // 3].tolist() \
            + (fin // 7 + paddle.randperm(fin // 7 * 2)[:fout // 3]).tolist() \
            + (fin // 7 * 3 + paddle.randperm(fin // 7 * 4)[:(fout - fout // 3 * 2)]).tolist()
        W = paddle.eye(fin)[s].T
    elif 'ortho' == method:
        W = orthogonal(fin, fout)
    elif 'svd_ortho' == method:
        W = svd_orthogonal(fin, fout)
    elif 'gaussian' == method:
        W = paddle.randn(fin, fout)
    return W


class PaDiMPlus(nn.Layer):
    def __init__(self, arch='resnet18', pretrained=True, k=10, method='sample'):
        super().__init__()
        if isinstance(arch, type(None)) or isinstance(pretrained, type(None)):
            self.model = None
            print('Inference mode')
        else:
            assert arch in models.keys(), 'arch {} not supported'.format(arch)
            self.model = models[arch](pretrained)
            del self.model.layer4, self.model.fc, self.model.avgpool
            self.model.eval()
            print(
                f'model {arch}, nParams {sum([w.size for w in self.model.parameters()])}'
            )
            self.arch = arch
            self.method = method
            self.fin = fins[arch]
            self.k = k
            self.projection = None
            self.reset_stats()

    def init_projection(self):
        self.projection = get_projection(fins[self.arch], self.k, self.method)

    def load(self, state):
        self.mean = state['mean']
        self.inv_covariance = state['inv_covariance']
        self.projection = state['projection']

    def reset_stats(self, set_None=True):
        if set_None:
            self.mean = None
            self.inv_covariance = None
        else:
            self.mean = paddle.zeros_like(self.mean)
            self.inv_covariance = paddle.zeros_like(self.inv_covariance)

    def set_dist_params(self, mean, inv_cov):
        self.mean, self.inv_covariance = mean, inv_cov

    @paddle.no_grad()
    def project_einsum(self, x):
        return paddle.einsum('bchw, cd -> bdhw', x, self.projection)
        # if self.method == 'ortho':
        #    return paddle.einsum('bchw, cd -> bdhw', x, self.projection)
        # else: #self.method == 'PaDiM':
        #    return paddle.index_select(embedding,  self.projection, 1)

    @paddle.no_grad()
    def project(self, x, return_HWBC=False):
        if isinstance(self.projection, type(None)):
            return x.transpose((2, 3, 0, 1)) if return_HWBC else x
        B, C, H, W = x.shape
        if len(self.projection.shape) == 1:
            x = paddle.index_select(x, self.projection, 1)
            if return_HWBC: x = x.transpose((2, 3, 0, 1))
            return x
        else:
            if return_HWBC:
                x = x.transpose((2, 3, 0, 1))
                return x @self.projection
                result = []  # paddle.zeros((B, self.k, H, W))
                for i in range(H):
                    # result[i] = paddle.einsum('chw, cd -> dhw', x[i], self.projection)
                    # result[i,:,:,:] = x[i] @self.projection.T
                    result.append(x[i] @self.projection.T)
                result = paddle.stack(result)
                return result
            result = []  # paddle.zeros((B, self.k, H, W))
            x = x.reshape((B, C, H * W))
            for i in range(B):
                # result[i] = paddle.einsum('chw, cd -> dhw', x[i], self.projection)
                # result[i] = (self.projection.T @ x[i]).reshape((self.k, H, W))
                result.append((self.projection.T @x[i]).reshape((self.k, H, W)))
            result = paddle.stack(result)
            return result

    @paddle.no_grad()
    def forward_res(self, x):
        res = []
        with paddle.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res

    @paddle.no_grad()
    def forward(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        res.append(x)
        x = self.model.layer2(x)
        res.append(x)
        x = self.model.layer3(x)
        res.append(x)
        x = res
        for i in range(1, len(x)):
            x[i] = F.interpolate(x[i], scale_factor=2**i, mode="nearest")
        # print([i.shape for i in x])
        x = paddle.concat(x, 1)
        # x = self.project(x)
        return x

    @paddle.no_grad()
    def forward_score(self, x):
        return self.generate_scores_map(self.get_embedding(x), x.shape)

    @paddle.no_grad()
    def compute_stats_einsum(self, outs):
        # calculate multivariate Gaussian distribution
        B, C, H, W = outs.shape
        mean = outs.mean(0)  # mean chw
        outs -= mean
        cov = paddle.einsum('bchw, bdhw -> hwcd', outs, outs) / (
            B - 1)  # covariance hwcc
        self.compute_inv(mean, cov)

    @paddle.no_grad()
    def compute_stats_(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W))
        cov = np.empty((C, C, H * W))
        for i in tqdm(range(H * W)):
            cov[:, :, i] = np.cov(embedding[:, :, i].numpy(), rowvar=False)
        cov = paddle.to_tensor(cov.reshape(C, C, H, W).transpose((2, 3, 0, 1)))
        self.compute_inv(mean, cov)

    @paddle.no_grad()
    def compute_stats_np(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W)).numpy()
        inv_covariance = np.empty((H * W, C, C), dtype='float32')
        I = np.identity(C)
        for i in tqdm(range(H * W)):
            inv_covariance[i, :, :] = np.linalg.inv(
                np.cov(embedding[:, :, i], rowvar=False) + 0.01 * I)
        inv_covariance = paddle.to_tensor(inv_covariance.reshape(
            H, W, C, C)).astype('float32')
        self.set_dist_params(mean, inv_covariance)

    @paddle.no_grad()
    def compute_stats(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding -= mean
        embedding = embedding.transpose((2, 3, 0, 1))  # hwbc
        inv_covariance = []  # paddle.zeros((H, W, C, C), dtype='float32')
        I = paddle.eye(C)
        for i in tqdm(range(H), desc='compute distribution stats'):
            inv_covariance.append(
                paddle.einsum('wbc, wbd -> wcd', embedding[i], embedding[i]) / (
                    B - 1) + 0.01 * I)
            inv_covariance[-1] = cholesky_inverse(
                inv_covariance[-1])  # paddle.inverse(inv_covariance[-1])#
        inv_covariance = paddle.stack(inv_covariance).reshape(
            (H, W, C, C)).astype('float32')
        self.set_dist_params(mean, inv_covariance)

    @paddle.no_grad()
    def compute_stats_incremental(self, out):
        # calculate multivariate Gaussian distribution
        H, W, B, C = out.shape
        if isinstance(self.inv_covariance, type(None)):
            self.mean = paddle.zeros((H, W, C), dtype='float32')
            self.inv_covariance = paddle.zeros((H, W, C, C), dtype='float32')

        self.mean += out.sum(2)  # mean hwc
        # cov = paddle.einsum('bchw, bdhw -> hwcd', outs, outs)# covariance hwcc
        for i in range(H):
            self.inv_covariance[i, :, :, :] += paddle.einsum(
                'wbc, wbd -> wcd', out[i, :, :, :], out[i, :, :, :])
        # return mean, cov, B

    def compute_inv_incremental(self, B, eps=0.01):
        c = self.mean.shape[0]
        # if self.inv_covariance == None:
        self.mean /= B  # hwc
        self.inv_covariance /= B
        # covariance hwcc  #.transpose((2,3, 0, 1)))
        self.inv_covariance -= paddle.einsum('hwc, hwd -> hwcd', self.mean,
                                             self.mean)
        # covariance = (covariance - B*paddle.einsum('chw, dhw -> hwcd', mean, mean))/(B-1)
        self.compute_inv(
            self.mean.transpose((2, 0, 1)), self.inv_covariance, eps)

    def compute_inv_(self, mean, covariance, eps=0.01):
        c = mean.shape[0]
        # if self.inv_covariance == None:
        # covariance hwcc  #.transpose((2,3, 0, 1)))
        # self.inv_covariance = paddle.linalg.inv(covariance)
        self.set_dist_params(mean,
                             cholesky_inverse(covariance + eps * paddle.eye(c)))

    def compute_inv(self, mean, covariance, eps=0.01):
        c, H, W = mean.shape
        for i in tqdm(range(H), desc='compute inverse covariance'):
            covariance[i, :, :, :] = cholesky_inverse(covariance[i, :, :, :] +
                                                      eps * paddle.eye(c))
        self.set_dist_params(mean, covariance)

    def generate_scores_map(self, embedding, out_shape, gaussian_blur=True):
        # calculate distance matrix
        # B, C, H, W = embedding.shape
        # embedding = embedding.reshape((B, C, H * W))

        # calculate mahalanobis distances
        distances = mahalanobis_einsum(embedding, self.mean,
                                       self.inv_covariance)
        score_map = postporcess_score_map(distances, out_shape, gaussian_blur)
        img_score = score_map.reshape(score_map.shape[0], -1).max(axis=1)
        return score_map, img_score
        return


@register
class PatchCore(PaDiMPlus):
    def load(self, state):
        self.memory_bank = state['memory_bank']

    def clean_stats(self, set_None=True):
        if set_None:
            self.memory_bank = None
        else:
            self.memory_bank = paddle.zeros_like(self.memory_bank)

    def set_dist_params(self, memory_bank):
        self.memory_bank = memory_bank

    @paddle.no_grad()
    def forward_res(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        x = self.model.layer3(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        return res

    @paddle.no_grad()
    def forward(self, x):
        pool = paddle.nn.AvgPool2D(3, 1, 1, exclusive=False)
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        res.append(pool(x))
        x = self.model.layer3(x)
        res.append(pool(x))
        x = res
        for i in range(1, len(x)):
            x[i] = F.interpolate(x[i], scale_factor=2**i, mode="nearest")
        # print([i.shape for i in x])
        x = paddle.concat(x, 1)
        # x = self.project(x)
        return x

    @paddle.no_grad()
    def compute_stats(self, embedding):
        C = embedding.shape[1]
        embedding = embedding.transpose((0, 2, 3, 1)).reshape((-1, C))
        print("Creating CoreSet Sampler via k-Center Greedy")
        sampler = KCenterGreedy(embedding, sampling_ratio=self.k / 100)
        print("Getting the coreset from the main embedding.")
        coreset = sampler.sample_coreset()
        print(
            f"Assigning the coreset as the memory bank with shape {coreset.shape}."
        )  # 18032,384
        self.memory_bank = coreset

    def compute_stats_einsum(self, outs):
        raise NotImplementedError

    def compute_stats_incremental(self, out):
        raise NotImplementedError

    def compute_inv_incremental(self, B, eps=0.01):
        raise NotImplementedError

    def project(self, x, return_HWBC=False):
        # no per project
        return x  # super().project(x, return_HWBC)

    def generate_scores_map(self, embedding, out_shape, gaussian_blur=True):
        # Nearest Neighbours distances
        B, C, H, W = embedding.shape
        embedding = embedding.transpose((0, 2, 3, 1)).reshape((B, H * W, C))
        distances = self.nearest_neighbors(embedding=embedding, n_neighbors=9)
        distances = distances.transpose((2, 0, 1))  # n_neighbors, B, HW
        image_score = []
        for i in range(B):
            image_score.append(
                self.compute_image_anomaly_score(distances[:, i, :]))
        distances = distances[0, :, :].reshape((B, H, W))
        score_map = postporcess_score_map(distances, out_shape, gaussian_blur)
        return score_map, np.array(image_score)

    def nearest_neighbors(self, embedding, n_neighbors: int=9):
        """Compare embedding Features with the memory bank to get Nearest Neighbours distance
        """
        B, HW, C = embedding.shape
        n_coreset = self.memory_bank.shape[0]
        distances = []  # paddle.zeros((B, HW, n_coreset))
        for i in range(B):
            distances.append(
                cdist(
                    embedding[i, :, :], self.memory_bank,
                    p=2.0))  # euclidean norm
        distances = paddle.stack(distances, 0)
        distances, _ = distances.topk(k=n_neighbors, axis=-1, largest=False)
        return distances  # B,

    @staticmethod
    def compute_image_anomaly_score(distance):
        """Compute Image-Level Anomaly Score for one nearest_neighbor distance map.
        """
        # distances[n_neighbors, B, HW]
        max_scores = paddle.argmax(distance[0, :])
        confidence = distance[:,
                              max_scores]  # paddle.index_select(distances, max_scores, -1)
        weights = 1 - (paddle.max(paddle.exp(confidence)) /
                       paddle.sum(paddle.exp(confidence)))
        score = weights * paddle.max(distance[0, :])
        return score.item()


def postporcess_score_map(distances,
                          out_shape,
                          gaussian_blur=True,
                          mode='bilinear'):
    score_map = F.interpolate(
        distances.unsqueeze_(1), size=out_shape, mode=mode,
        align_corners=False).squeeze_(1).numpy()
    if gaussian_blur:
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map


def get_model(method):
    if 'coreset' == method:
        return PatchCore
    return PaDiMPlus


if __name__ == '__main__':
    model = PaDiMPlus()
    print(model)
