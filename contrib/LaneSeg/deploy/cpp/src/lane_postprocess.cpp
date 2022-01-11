// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lane_postprocess.hpp"

LanePostProcess::LanePostProcess(int src_height, int src_width, int rows, int cols, int num_classes):num_classes(num_classes) {
    src_height = src_height;
    src_width = src_width;
    input_height = rows;
    input_width = cols;
}

vector<int> LanePostProcess::get_lane(cv::Mat& prob_map) {
    int dst_height = src_height - cut_height;
    int pointCount = 0;
    vector<int> coords(points_nums, -1);
    for(int k = 0; k < points_nums; ++k) {
        int y = int((dst_height - 10 - k * y_pixel_gap) * input_height / dst_height);
        if (y < 0)
            break;
        const float* p = prob_map.ptr<float>(y);
        int id = max_element(p, p + prob_map.cols) - p;
        float val = *max_element(p, p + prob_map.cols);
        if (val > thresh) {
            coords[k] = int(id * 1.0 / input_width * src_width);
            pointCount++;
        }
    }
    if (pointCount < 2) {
        std::fill(coords.begin(), coords.end(), 0);
    }
    process_gap(coords);
    return coords;
}

void LanePostProcess::process_gap(vector<int>& coordinate) {
    int start = -1;
    int end = -1;
    int len = coordinate.size();
    for(int left = 0; left < len; ++left) {
        if (coordinate[left] != -1) {
            start = left;
            break;
        }
    }

    for(int right = len - 1; right > 0; --right) {
        if (coordinate[right] != -1) {
            end = right;
            break;
        }
    }

    if (start == -1 || end == -1) {
        return;
    }

    vector<int> lane;
    lane.assign(coordinate.begin() + start, coordinate.begin() + end + 1);

    bool has_gap = false;
    for(int i = 0; i < lane.size(); ++i) {
        if (lane[i] == -1) {
            has_gap = true;
        }
    }
    if (has_gap) {
        vector<int> gap_start;
        vector<int> gap_end;
        int lane_sz =lane.size();
        for (int iter = 0; iter < lane_sz; iter++) {
            if(iter+1 < lane_sz && lane[iter] != -1 && lane[iter+1] == -1) {
                gap_start.push_back(iter);
            }

            if (iter+1 < lane_sz && lane[iter] == -1 && lane[iter+1] != -1) {
                gap_end.push_back(iter + 1);
            }
        }

        if (gap_start.size() == 0 || gap_end.size() == 0)
            return;

        for(int g = 0; g < gap_start.size(); g++) {
            if (g >= gap_end.size()) {
                return ;
            }
            for(int id = gap_start[g] + 1; id < gap_end[g]; id++) {
                float gap_width = float(gap_end[g] - gap_start[g]);
                lane[id] = int((id - gap_start[g]) / gap_width * lane[gap_end[g]] +
                               (gap_end[g] - id) / gap_width * lane[gap_start[g]]);
            }
        }
        copy(lane.begin(), lane.end(), coordinate.begin() + start);
    }
}

vector<vector<pair<int, int>>> LanePostProcess::heatmap2lane(int num_classes, cv::Mat seg_planes[]) {
    vector<vector<pair<int, int>>> coordinates;
    vector<int> coords(points_nums, -1);
    for(int i = 0; i < num_classes - 1; i++) {
        cv::Mat prob_map = seg_planes[i+1];
        if (true) {
            cv::blur(prob_map, prob_map, Size(9, 9), Point(-1,-1), BORDER_REPLICATE );
        }
        coords = get_lane(prob_map);
        int sum = accumulate(coords.begin(), coords.end(), 0);

        if (sum == 0) {
            continue;
        }
         add_coords(coordinates, coords);

    }
    if (coordinates.size() == 0) {
        std::fill(coords.begin(), coords.end(), 0);
        add_coords(coordinates, coords);
    }
    return coordinates;
}

vector<vector<pair<int, int>>> LanePostProcess::lane_process(vector<float>& out_data, int cut_height)
{
    cut_height = cut_height;
    cv::Size size = cv::Size(input_width, input_height);
    cv::Mat softmax_mat;
    softmax_mat.create(size, CV_32FC(num_classes));

    vector<float> src(num_classes, 0);
    vector<float> dst(num_classes, 0);

    for (int i = 0; i < input_height; ++i) {
        for(int j = 0; j < input_width; ++j) {
            for(int idx = 0; idx < num_classes; ++idx) {
                src[idx] = out_data[i * input_width + j + input_height * input_width * idx];
            }
            softmax(&src[0], &dst[0], num_classes);
            for(int idx = 0; idx < num_classes; ++idx) {
                softmax_mat.at<cv::Vec<float, 7>>(i,j)[idx] = dst[idx];
            }
        }
    }

    cv::Mat seg_planes[num_classes];
    for(int i = 0; i < num_classes; i++) {
        seg_planes[i].create(size, CV_32FC(1));
    }

    cv::split(softmax_mat, seg_planes);
    auto lane_coords = heatmap2lane(num_classes, seg_planes);
    return lane_coords;
}

void LanePostProcess::add_coords(vector<vector<pair<int, int>>>& coordinates, vector<int>& coords) {
    vector<pair<int, int>> sub_lanes;
    for (int j = 0; j < points_nums; ++j) {
        if (coords[j] > 0) {
            sub_lanes.push_back({coords[j], src_height - 10 - j * y_pixel_gap});
        } else {
            sub_lanes.push_back({-1, src_height - 10 - j * y_pixel_gap});
        }
    }
    coordinates.push_back(sub_lanes);
}

void LanePostProcess::softmax(float* src, float* dst, int length) {
    float sum = 0.0f;
    for(int i = 0; i < length; ++i) {
        dst[i] = exp(src[i]);
        sum += dst[i];
    }
    for(int i = 0; i <length; ++i) {
        dst[i] = dst[i] / sum;
    }
}
