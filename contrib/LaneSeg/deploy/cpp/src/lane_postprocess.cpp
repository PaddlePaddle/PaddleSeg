//
//  lane_postprocess.cpp
//  test_seg
//
//  Created by Huang,Shenghui on 2021/11/11.
//

#include "lane_postprocess.hpp"

LanePostProcess::LanePostProcess() {

}

vector<int> LanePostProcess::get_lane(cv::Mat& prob_map, int y_px_gap, int pts, float thresh,
              int H, int W, int h, int w, int cut_height) {
    H -= cut_height;
    int pointCount = 0;
    vector<int> coords(pts, -1);
    for(int k = 0; k < pts; ++k) {
        int y = int((H - 10 - k * y_px_gap) * h / H);
        if (y < 0)
            break;
        const float* p = prob_map.ptr<float>(y);
        int id = max_element(p, p + prob_map.cols) - p;
        float val = *max_element(p, p + prob_map.cols);
        if (val > thresh) {
            coords[k] = int(id * 1.0 / w * W);
            pointCount++;
        }
    }
    if (pointCount < 2) {
        std::fill(coords.begin(), coords.end(), 0);
    }
    fix_gap(coords);
    return coords;
}

void LanePostProcess::fix_gap(vector<int>& coordinate) {
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

vector<vector<pair<int, int>>> LanePostProcess::probmap2lane(int num_classes, cv::Mat seg_planes[]) {
    int y_px_gap = 10;
    int pts = 56;
    float thresh = 0.6;
    int H = 720, W = 1280;
    int h = 368, w = 640;
    int cut_height = 160;
    vector<vector<pair<int, int>>> coordinates;
    vector<int> coords(pts, -1);
    for(int i = 0; i < num_classes - 1; i++) {
        cv::Mat prob_map = seg_planes[i+1];
        if (true) {
            cv::blur(prob_map, prob_map, Size(9, 9), Point(-1,-1), BORDER_REPLICATE );
        }
        coords = get_lane(prob_map,  y_px_gap,  pts, thresh, H,  W,  h,  w, cut_height);
        int sum = accumulate(coords.begin(), coords.end(), 0);

        if (sum == 0) {
            continue;
        }
         add_coords( coordinates, coords,  H, y_px_gap);

    }
    if (coordinates.size() == 0) {
        std::fill(coords.begin(), coords.end(), 0);
        add_coords( coordinates, coords, H,  y_px_gap);
    }
    return coordinates;
}

vector<vector<pair<int, int>>> LanePostProcess::lane_process(int num_classes, cv::Size size,
                  int out_num, int skip_index)
{
    vector<float> seg_pred(out_num, 0);
    std::ifstream ifs("", std::ios::binary | std::ios::in);
    ifs.read(reinterpret_cast<char *>(&seg_pred[0]), sizeof(float) * out_num);
    ifs.close();

    cv::Mat seg_planes[num_classes];
    for(int i = 0; i < num_classes; i++) {
        seg_planes[i].create(size, CV_32FC(1));
    }

    for(int i = 0; i < num_classes; i++) {
        ::memcpy(seg_planes[i].data, seg_pred.data() + i *skip_index, skip_index * sizeof(float)); //内存拷贝
    }
    auto lane_coords = probmap2lane(num_classes, seg_planes);
    return lane_coords;
}

void LanePostProcess::add_coords(vector<vector<pair<int, int>>> & coordinates, vector<int> & coords, int H, int y_px_gap) {
    vector<pair<int, int>> sub_lanes;
    for (int j = 0; j < 56; ++j) {
        if (coords[j] > 0) {
            sub_lanes.push_back({coords[j], H - 10 - j * y_px_gap});
        } else {
            sub_lanes.push_back({-1, H - 10 - j * y_px_gap});
        }
    }
    coordinates.push_back(sub_lanes);
}
