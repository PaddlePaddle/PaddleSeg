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

#ifndef lane_postprocess_hpp
#define lane_postprocess_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include<numeric>

using namespace::std;
using namespace::cv;

class LanePostProcess {

public:
    LanePostProcess(int src_height, int src_width, int rows, int cols, int num_classes);

    vector<int> get_lane(cv::Mat& prob_map);

    void process_gap(vector<int>& coordinate);

    vector<vector<pair<int, int>>> heatmap2lane(int num_classes, cv::Mat seg_planes[]);

    vector<vector<pair<int, int>>> lane_process(vector<float>& out_data, int cut_height);

    void add_coords(vector<vector<pair<int, int>>> & coordinates, vector<int>& coords);

    void softmax(float* src, float* dst, int length);

    ~LanePostProcess(){}

private:
    int cut_height = 160;
    int src_height = 720;
    int src_width = 1280;

    int num_classes = 7;
    int input_height = 368;
    int input_width = 640;

    int y_pixel_gap = 10;
    int points_nums = 56;
    float thresh = 0.6;

};

#endif /* lane_postprocess_hpp */
