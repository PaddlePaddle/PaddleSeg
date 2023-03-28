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

import warnings
from typing import Optional

import paddle
from paddle import Tensor
import numpy as np
from sklearn.utils.random import sample_without_replacement


class NotFittedError(ValueError, AttributeError):
    """Raise Exception if estimator is used before fitting."""


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float=0.1):
    """Find a 'safe' number of components to randomly project to.

    Ref eqn 2.1 https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf

    Args:
        n_samples (int): Number of samples used to compute safe components
        eps (float, optional): Minimum distortion rate. Defaults to 0.1.
    """

    denominator = (eps**2 / 2) - (eps**3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)


class BaseRandomProjection:
    """Base module for Random Projection using Paddle operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def __init__(self,
                 n_components="auto",
                 eps: float=0.1,
                 random_state: Optional[int]=None) -> None:
        self.n_components = n_components
        self.n_components_: int
        self.random_matrix: Tensor
        self.eps = eps
        self.random_state = random_state

    def _make_random_matrix(self, n_components: int, n_features: int):
        """Random projection matrix.

        Args:
            n_components (int): Dimensionality of the target projection space.
            n_features (int): Dimentionality of the original source space.

        Returns:
            Tensor: Matrix of shape (n_components, n_features).
        """
        raise NotImplementedError

    def fit(self, embedding: Tensor) -> "BaseRandomProjection":
        """Generates projection matrix from the embedding tensor.

        Args:
            embedding (Tensor): embedding tensor for generating embedding

        Returns:
            (RandomProjection): Return self to be used as
            > generator = RandomProjection()
            > generator = generator.fit()
        """
        n_samples, n_features = embedding.shape
        # device = embedding.device
        # ported from sklearn
        if self.n_components == "auto":
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps)

            if self.n_components_ <= 0:
                raise ValueError(
                    "eps=%f and n_samples=%d lead to a target dimension of "
                    "%d which is invalid" %
                    (self.eps, n_samples, self.n_components_))

            elif self.n_components_ > n_features:
                raise ValueError(
                    "eps=%f and n_samples=%d lead to a target dimension of "
                    "%d which is larger than the original space with "
                    "n_features=%d" %
                    (self.eps, n_samples, self.n_components_, n_features))
        else:
            if self.n_components <= 0:
                raise ValueError("n_components must be greater than 0, got %s" %
                                 self.n_components)

            elif self.n_components > n_features:
                warnings.warn(
                    "The number of components is higher than the number of"
                    " features: n_features < n_components (%s < %s)."
                    "The dimensionality of the problem will not be reduced." %
                    (n_features, self.n_components))

            self.n_components_ = self.n_components

        # Generate projection matrix
        self.random_matrix = self._make_random_matrix(
            self.n_components_, n_features=n_features)  # .to(device)

        return self

    def transform(self, embedding: Tensor) -> Tensor:
        """Project the data by using matrix product with the random matrix.

        Args:
            embedding (Tensor): Embedding of shape (n_samples, n_features)
                The input data to project into a smaller dimensional space

        Returns:
            projected_embedding (Tensor): Matrix of shape
                (n_samples, n_components) Projected array.
        """
        if self.random_matrix is None:
            raise NotFittedError(
                "`fit()` has not been called on RandomProjection yet.")

        projected_embedding = embedding @self.random_matrix.T
        return projected_embedding


class SparseRandomProjection(BaseRandomProjection):
    """Sparse Random Projection using Paddle operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def _make_random_matrix(self, n_components: int, n_features: int):
        """Random sparse matrix. Based on https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf.

        # paddle can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix

        Args:
            n_components (int): Dimensionality of the target projection space.
            n_features (int): Dimentionality of the original source space.

        Returns:
            Tensor: Sparse matrix of shape (n_components, n_features).
                The generated Gaussian random matrix is in CSR (compressed sparse row)
                format.
        """

        # Density 'auto'. Factorize density
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = paddle.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(n_components) * components

        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = paddle.zeros(
                (n_components, n_features), dtype=paddle.float64)
            for i in range(n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = np.random.binomial(n_features, density)
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = paddle.to_tensor(
                    sample_without_replacement(
                        n_population=n_features,
                        n_samples=nnz_idx,
                        random_state=self.random_state),
                    dtype='int64', )
                data = paddle.to_tensor(
                    np.random.binomial(1, 0.5, c_idx.size)) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data

            components *= np.sqrt(1 / density) / np.sqrt(n_components)

        return components.astype('float32')

    def fit(self, embedding: Tensor) -> "SparseRandomProjection":
        """Generates sparse matrix from the embedding tensor.

        Args:
            embedding (Tensor): embedding tensor for generating embedding

        Returns:
            (SparseRandomProjection): Return self to be used as
            >>> generator = SparseRandomProjection()
            >>> generator = generator.fit()
        """
        return super().fit(embedding)

    def transform(self, embedding: Tensor) -> Tensor:
        """Project the data by using matrix product with the random matrix.

        Args:
            embedding (Tensor): Embedding of shape (n_samples, n_features)
                The input data to project into a smaller dimensional space

        Returns:
            projected_embedding (Tensor): Sparse matrix of shape
                (n_samples, n_components) Projected array.
        """
        return super().transform(embedding)


class GaussianProjection(BaseRandomProjection):
    """Gaussian Random Projection using Paddle operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def _make_random_matrix(self, n_components: int, n_features: int):
        components = paddle.Tensor(n_components, n_features)
        paddle.nn.init.normal_(components)
        return components


class SemiOrthoProjection(BaseRandomProjection):
    """Semi Orthogonal Random Projection using Paddle operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def _make_random_matrix(self, n_components: int, n_features: int):
        components = paddle.Tensor(n_components, n_features)
        paddle.nn.init.orthogonal_(components)
        return components
