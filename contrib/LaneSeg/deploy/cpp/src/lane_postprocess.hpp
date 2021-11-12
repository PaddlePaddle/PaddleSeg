//
//  lane_postprocess.hpp
//  test_seg
//
//  Created by Huang,Shenghui on 2021/11/11.
//

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
