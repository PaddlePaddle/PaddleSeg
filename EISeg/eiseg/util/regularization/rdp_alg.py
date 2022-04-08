# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/niecongchong/RS-building-regularization
Ths copyright of niecongchong/RS-building-regularization is as follows:
Apache License [see LICENSE for details]
"""
"""
rdp
~~~
Pure Python implementation of the Ramer-Douglas-Peucker algorithm.
:copyright: (c) 2014 Fabian Hirschmann <fabian@hirschmann.email>
:license: MIT, see LICENSE.txt for more details.
"""

import numpy as np


def pldist(x0, x1, x2):
    """
    Calculates the distance from the point ``x0`` to the line given
    by the points ``x1`` and ``x2``.
    :param x0: a point
    :type x0: a 2x1 numpy array
    :param x1: a point of the line
    :type x1: 2x1 numpy array
    :param x2: another point of the line
    :type x2: 2x1 numpy array
    """
    x0, x1, x2 = x0[:2], x1[:2], x2[:2]  # discard timestamp
    if x1[0] == x2[0]:
        return np.abs(x0[0] - x1[0])
    return np.divide(
        np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
        np.linalg.norm(x2 - x1))


def _rdp(M, epsilon, dist):
    """
    Simplifies a given array of points.
    :param M: an array
    :type M: Nx2 numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(x1, x2, x3)``
    """
    dmax = 0.0
    index = -1
    for i in range(1, M.shape[0]):
        d = dist(M[i], M[0], M[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        r1 = _rdp(M[:index + 1], epsilon, dist)
        r2 = _rdp(M[index:], epsilon, dist)
        return np.vstack((r1[:-1], r2))
    else:
        return np.vstack((M[0], M[-1]))


def _rdp_nn(seq, epsilon, dist):
    """
    Simplifies a given array of points.
    :param seq: a series of points
    :type seq: sequence of 2-tuples
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(x1, x2, x3)``
    """
    return _rdp(np.array(seq), epsilon, dist).tolist()


def rdp(M, epsilon=0, dist=pldist):
    """
    Simplifies a given array of points.
    :param M: a series of points
    :type M: either a Nx2 numpy array or sequence of 2-tuples
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(x1, x2, x3)``
    """
    if "numpy" in str(type(M)):
        return _rdp(M, epsilon, dist)
    return _rdp_nn(M, epsilon, dist)
