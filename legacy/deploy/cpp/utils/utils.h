// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef _WIN32
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

namespace PaddleSolution {
namespace utils {
    inline std::string path_join(const std::string& dir,
                                 const std::string& path) {
        std::string seperator = "/";
        #ifdef _WIN32
        seperator = "\\";
        #endif
        return dir + seperator + path;
    }
    #ifndef _WIN32
    // scan a directory and get all files with input extensions
    inline std::vector<std::string> get_directory_images(
                        const std::string& path, const std::string& exts) {
        std::vector<std::string> imgs;
        struct dirent *entry;
        DIR *dir = opendir(path.c_str());
        if (dir == NULL) {
            closedir(dir);
            return imgs;
        }

        while ((entry = readdir(dir)) != NULL) {
            std::string item = entry->d_name;
            auto ext = strrchr(entry->d_name, '.');
            if (!ext || std::string(ext) == "." || std::string(ext) == "..") {
                continue;
            }
            if (exts.find(ext) != std::string::npos) {
                imgs.push_back(path_join(path, entry->d_name));
            }
        }
        return imgs;
    }
    #else
    // scan a directory and get all files with input extensions
    inline std::vector<std::string> get_directory_images(
                    const std::string& path, const std::string& exts) {
        std::string pattern(path);
        pattern.append("\\*");
        std::vector<std::string> imgs;
        WIN32_FIND_DATA data;
        HANDLE hFind;
        if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
            do {
                auto fname = std::string(data.cFileName);
                auto pos = fname.rfind(".");
                auto ext = fname.substr(pos + 1);
                if (ext.size() > 1 && exts.find(ext) != std::string::npos) {
                    imgs.push_back(path + "\\" + data.cFileName);
                }
            } while (FindNextFile(hFind, &data) != 0);
            FindClose(hFind);
        }
        return imgs;
    }
    #endif

    // normalize and HWC_BGR -> CHW_RGB
    inline void normalize(cv::Mat& im, float* data, std::vector<float>& fmean,
                          std::vector<float>& fstd) {
        int rh = im.rows;
        int rw = im.cols;
        int rc = im.channels();
        double normf = static_cast<double>(1.0) / 255.0;
        #pragma omp parallel for
        for (int h = 0; h < rh; ++h) {
            const uchar* ptr = im.ptr<uchar>(h);
            int im_index = 0;
            for (int w = 0; w < rw; ++w) {
                for (int c = 0; c < rc; ++c) {
                    int top_index = (c * rh + h) * rw + w;
                    float pixel = static_cast<float>(ptr[im_index++]);
                    pixel = (pixel * normf - fmean[c]) / fstd[c];
                    data[top_index] = pixel;
                }
            }
        }
    }

    // flatten a cv::mat
    inline void flatten_mat(cv::Mat& im, float* data) {
        int rh = im.rows;
        int rw = im.cols;
        int rc = im.channels();
        #pragma omp parallel for
        for (int h = 0; h < rh; ++h) {
            const uchar* ptr = im.ptr<uchar>(h);
            int im_index = 0;
            int top_index = h * rw * rc;
            for (int w = 0; w < rw; ++w) {
                for (int c = 0; c < rc; ++c) {
                    float pixel = static_cast<float>(ptr[im_index++]);
                    data[top_index++] = pixel;
                }
            }
        }
    }

    // argmax
    inline void argmax(float* out, std::vector<int>& shape,
                       std::vector<uchar>& mask, std::vector<uchar>& scoremap) {
        int out_img_len = shape[1] * shape[2];
        int blob_out_len = out_img_len * shape[0];
        float max_value = -1;
        int label = 0;
        #pragma omp parallel private(label)
        for (int i = 0; i < out_img_len; ++i) {
            max_value = -1;
            label = 0;
            #pragma omp for reduction(max : max_value)
            for (int j = 0; j < shape[0]; ++j) {
                int index = i + j * out_img_len;
                if (index >= blob_out_len) {
                    continue;
                }
                float value = out[index];
                if (value > max_value) {
                    max_value = value;
                    label = j;
                }
            }
            if (label == 0) max_value = 0;
            mask[i] = uchar(label);
            scoremap[i] = uchar(max_value * 255);
        }
    }
}  // namespace utils
}  // namespace PaddleSolution
