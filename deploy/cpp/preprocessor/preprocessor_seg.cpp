// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "preprocessor_seg.h"

#include <glog/logging.h>

#include <thread>


namespace PaddleSolution {

    bool SegPreProcessor::single_process(const std::string& fname,
                                         float* data, int* ori_w, int* ori_h) {
        cv::Mat im = cv::imread(fname, -1);
        if (im.data == nullptr || im.empty()) {
            LOG(ERROR) << "Failed to open image: " << fname;
            return false;
        }
        int channels = im.channels();
        *ori_w = im.cols;
        *ori_h = im.rows;

        if (channels == 1) {
            cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
        }
        channels = im.channels();
        if (channels != 3 && channels != 4) {
            LOG(ERROR) << "Only support rgb(gray) and rgba image.";
            return false;
        }
        cv::Size resize_size(_config->_resize[0], _config->_resize[1]);
        int rw = resize_size.width;
        int rh = resize_size.height;
        if (*ori_h != rh || *ori_w != rw) {
            cv::resize(im, im, resize_size, 0, 0, cv::INTER_LINEAR);
        }

        if (!_config->_use_pr) {
            utils::normalize(im, data, _config->_mean, _config->_std);
        } else {
            utils::flatten_mat(im, data);
        }
        return true;
    }

    bool SegPreProcessor::batch_process(const std::vector<std::string>& imgs,
                                        float* data, int* ori_w, int* ori_h) {
        auto ic = _config->_channels;
        auto iw = _config->_resize[0];
        auto ih = _config->_resize[1];
        std::vector<std::thread> threads;
        for (int i = 0; i < imgs.size(); ++i) {
            std::string path = imgs[i];
            float* buffer = data + i * ic * iw * ih;
            int* width = &ori_w[i];
            int* height = &ori_h[i];
            threads.emplace_back([this, path, buffer, width, height] {
                single_process(path, buffer, width, height);
                });
        }
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        return true;
    }

    bool SegPreProcessor::init(
        std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> config) {
        _config = config;
        return true;
    }
}  // namespace PaddleSolution
