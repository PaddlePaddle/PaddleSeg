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
import os
import sys
import glob
import tarfile
import time
import zipfile
import functools
import requests
import shutil

lasttime = time.time()
FLUSH_INTERVAL = 0.1


class uncompressor:
    def __init__(self, download_params):
        if download_params is not None:
            urls, savepath, print_progress = download_params
            for key, url in urls.items():
                if url:
                    self._download_file(
                        url,
                        savepath=os.path.join(savepath, key),
                        print_progress=print_progress)

    def _uncompress_file_zip(self, filepath, extrapath):
        files = zipfile.ZipFile(filepath, 'r')
        filelist = files.namelist()
        rootpath = filelist[0]
        total_num = len(filelist)
        for index, file in enumerate(filelist):
            files.extract(file, extrapath)
            yield total_num, index, rootpath
        files.close()
        yield total_num, index, rootpath

    def progress(self, str, end=False):
        global lasttime
        if end:
            str += "\n"
            lasttime = 0
        if time.time() - lasttime >= FLUSH_INTERVAL:
            sys.stdout.write("\r%s" % str)
            lasttime = time.time()
            sys.stdout.flush()

    def _uncompress_file_tar(self, filepath, extrapath, mode="r:gz"):
        files = tarfile.open(filepath, mode)
        filelist = files.getnames()
        total_num = len(filelist)
        rootpath = filelist[0]
        for index, file in enumerate(filelist):
            files.extract(file, extrapath)
            yield total_num, index, rootpath
        files.close()
        yield total_num, index, rootpath

    def _uncompress_file(self, filepath, extrapath, delete_file,
                         print_progress):
        if print_progress:
            print("Uncompress %s" % os.path.basename(filepath))

        if filepath.endswith("zip"):
            handler = self._uncompress_file_zip
        elif filepath.endswith(("tgz", "tar", "tar.gz")):
            handler = functools.partial(self._uncompress_file_tar, mode="r:*")
        else:
            handler = functools.partial(self._uncompress_file_tar, mode="r")

        for total_num, index, rootpath in handler(filepath, extrapath):
            if print_progress:
                done = int(50 * float(index) / total_num)
                self.progress("[%-50s] %.2f%%" %
                              ('=' * done, float(100 * index) / total_num))
        if print_progress:
            self.progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

        if delete_file:
            os.remove(filepath)

        return rootpath

    def _download_file(self, url, savepath, print_progress):
        if print_progress:
            print("Connecting to {}".format(url))
        r = requests.get(url, stream=True, timeout=15)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(savepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            total_length = int(total_length)
            if os.path.exists(savepath) and total_length == os.path.getsize(
                    savepath):
                print("{} already downloaded, skipping".format(
                    os.path.basename(savepath)))
                return
            with open(savepath, 'wb') as f:
                dl = 0
                total_length = int(total_length)
                starttime = time.time()
                if print_progress:
                    print("Downloading %s" % os.path.basename(savepath))
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    if print_progress:
                        done = int(50 * dl / total_length)
                        self.progress(
                            "[%-50s] %.2f%%" %
                            ('=' * done, float(100 * dl) / total_length))
            if print_progress:
                self.progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)
