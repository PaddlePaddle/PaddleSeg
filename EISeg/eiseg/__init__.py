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

import sys
import os
import os.path as osp
import logging
from datetime import datetime

from qtpy import QtCore
import cv2

__APPNAME__ = "EISeg"
__VERSION__ = "1.0.0"

pjpath = osp.dirname(osp.realpath(__file__))
sys.path.append(pjpath)

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

# log
settings = QtCore.QSettings(
    osp.join(pjpath, "config/setting.txt"), QtCore.QSettings.IniFormat)

logFolder = settings.value("logFolder")
logLevel = bool(settings.value("log"))
logDays = settings.value("logDays")

if logFolder is None or len(logFolder) == 0:
    logFolder = osp.normcase(osp.join(pjpath, "log"))
if not osp.exists(logFolder):
    os.makedirs(logFolder)

if logLevel:
    logLevel = logging.DEBUG
else:
    logLevel = logging.CRITICAL
if logDays:
    logDays = int(logDays)
else:
    logDays = 7
# TODO: 删除大于logDays 的 log

t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("EISeg Logger")
handler = logging.FileHandler(
    osp.normcase(osp.join(logFolder, f"eiseg-{t}.log")))
handler.setFormatter(
    logging.Formatter(
        "%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - %(message)s"
    ))
logger.setLevel(logLevel)
logger.addHandler(handler)
