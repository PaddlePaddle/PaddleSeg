# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(LOCAL_PATH, "../../../", "test")
sys.path.append(TEST_PATH)

from paddleseg.utils.download import download_file_and_uncompress


def download_data(savepath, extrapath):
    url = "https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)

    url = "https://paddleseg.bj.bcebos.com/humanseg/data/video_test.zip"
    download_file_and_uncompress(
        url=url,
        savepath=savepath,
        extrapath=extrapath,
        extraname='video_test.mp4')


if __name__ == "__main__":
    download_data(LOCAL_PATH, LOCAL_PATH)
    print("Data download finish!")
