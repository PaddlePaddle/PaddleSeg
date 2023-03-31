# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from setuptools import find_packages, find_namespace_packages
from setuptools import setup

if __name__ == '__main__':
    with open("requirements.txt", 'r') as fr:
        REQUIRED_PACKAGES = fr.read()

    with open("paddlepanseg/.version", 'r') as fv:
        VERSION = fv.read().rstrip()

    setup(
        name='paddlepanseg',
        version=VERSION.replace('-', ''),
        description=(
            "End-to-end panoptic segmentation kit based on PaddleSeg."),
        long_description='',
        author='PaddlePaddle Authors',
        author_email='',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(include=['paddlepanseg', 'paddlepanseg.*']) +
        find_namespace_packages(include=['paddlepanseg', 'paddlepanseg.*']),
        include_package_data=True,
        # PyPI package information.
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords=('paddleseg paddlepaddle panoptic-segmentation'))
