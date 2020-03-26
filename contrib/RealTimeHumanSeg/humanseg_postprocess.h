//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

int ThresholdMask(const cv::Mat &fg_cfd,
                  const float fg_thres,
                  const float bg_thres,
                  cv::Mat fg_mask);

cv::Mat MergeSegMat(const cv::Mat& seg_mat,
                    const cv::Mat& ori_frame);

int MergeProcess(const uchar *im_buff,
                const float *im_scoremap_buff,
                const int height,
                const int width,
                uchar *result_buff);
