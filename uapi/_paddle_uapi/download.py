# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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
import os.path as osp
import sys
import shutil
import requests
import tqdm
import hashlib
import base64
import binascii

__all__ = ['get_path', ]

CONFIGS_HOME = osp.expanduser("~/.cache/paddlemodels/configs")

# dict of {dataset_name: (download_info, sub_dirs)}
# download info: [(url, md5sum)]

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') \
            or path.startswith('https://') \
            or path.startswith('paddlecv://')


def get_path(path):
    """Get config path from CONFIGS_HOME, if not exists,
    download it from url.
    """
    if not is_url(path):
        return path
    path, _ = get_path(path, CONFIGS_HOME)
    return path


def map_path(url, root_dir, path_depth=1):
    # parse path after download to decompress under root_dir
    assert path_depth > 0, "path_depth should be a positive integer"
    dirname = url
    for _ in range(path_depth):
        dirname = osp.dirname(dirname)
    fpath = osp.relpath(url, dirname)
    path = osp.join(root_dir, fpath)
    dirname = osp.dirname(path)
    return path, dirname


def get_path(url, root_dir, md5sum=None, check_exist=True, path_depth=1):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url, return the path.
    url (str): download url
    root_dir (str): root dir for downloading, it should be
                    WEIGHTS_HOME
    md5sum (str): md5 sum of download package
    """
    # parse path after download to decompress under root_dir
    fullpath, dirname = map_path(url, root_dir, path_depth)

    if osp.exists(fullpath) and check_exist:
        if not osp.isfile(fullpath) or \
                _check_exist_file_md5(fullpath, md5sum, url):
            return fullpath, True
        else:
            os.remove(fullpath)

    fullname = _download(url, dirname, md5sum)
    return fullpath, False


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _check_exist_file_md5(fullname, md5sum,
                                                              url)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        print("Downloading {} from {}".format(fname, url))

        # NOTE: windows path join may incur \, which is invalid in url
        if sys.platform == "win32":
            url = url.replace('\\', '/')

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
    return fullname


def _check_exist_file_md5(filename, md5sum, url):
    # if md5sum is None, and file to check is model file, 
    # read md5um from url and check, else check md5sum directly
    return _md5check_from_url(filename, url) if md5sum is None \
            and filename.endswith('pdparams') \
            else _md5check(filename, md5sum)


def _md5check_from_url(filename, url):
    # For model in bcebos URLs, MD5 value is contained
    # in request header as 'content_md5'
    req = requests.get(url, stream=True)
    content_md5 = req.headers.get('content-md5')
    req.close()
    if not content_md5 or _md5check(
            filename,
            binascii.hexlify(base64.b64decode(content_md5.strip('"'))).decode(
            )):
        return True
    else:
        return False


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        print("File {} md5 check failed, {}(calc) != "
              "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True
