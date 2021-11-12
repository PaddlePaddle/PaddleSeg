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
    LanePostProcess();

    vector<int> get_lane(cv::Mat& prob_map, int y_px_gap, int pts, float thresh,
                                          int H, int W, int h, int w, int cut_height);

    void fix_gap(vector<int>& coordinate);

    vector<vector<pair<int, int>>> probmap2lane(int num_classes, cv::Mat seg_planes[]);

    vector<vector<pair<int, int>>> lane_process(int num_classes = 7, cv::Size size = cv::Size(368, 640),
                                                                 int out_num = 2, int skip_index=2, vector<float> in = vector<float>());

    void add_coords(vector<vector<pair<int, int>>> & coordinates, vector<int> & coords, int H, int y_px_gap);
    ~LanePostProcess(){}
};

#endif /* lane_postprocess_hpp */
