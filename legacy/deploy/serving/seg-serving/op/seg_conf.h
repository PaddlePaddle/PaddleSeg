#ifndef SRC_SEG_CONF_H
#define SRC_SEG_CONF_H

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "gflags/gflags.h"

DECLARE_string(seg_conf_file);


namespace baidu{
namespace utils{
namespace seg_conf{
class SegConf{
private:
    explicit SegConf(const std::string &configuration_filename);
public:
    static const SegConf *instance();

    bool get_item_by_name(const std::string &conf_node_name, cv::FileNode &return_file_node) const;
    int get_mean_vector(std::vector<double> &mean_vec) const;
    int get_std_vector(std::vector<double> &std_vec) const;
    int get_size_vector(std::vector<int> &size_vec) const;
    int get_channels(int &channels) const;
    int get_class_num(int &class_num) const;
    int get_model_name(std::string &name) const;

    ~SegConf();
private:
    cv::FileStorage _seg_conf_file;
    //static SegConf s_seg_conf_instance;
    template <typename T>
    int get_array_from_file_node(const std::string &conf_node_name, std::vector<T> &vec) const{
        cv::FileNode node;
        if(!get_item_by_name(conf_node_name, node) && !node.isSeq()) {
            return -1;
        }
        //node >> vec;
        cv::FileNodeIterator start_file_node_iter = node.begin();
        cv::FileNodeIterator end_file_node_iter = node.end();
        for(cv::FileNodeIterator it = start_file_node_iter; it != end_file_node_iter; ++it) {
            vec.push_back(static_cast<T>(*it));
        }
        return 0;
    }

    template<typename T>
    int get_scalar_from_file_node(const std::string &conf_node_name, T &scalar) const{
        cv::FileNode node;
        if(!get_item_by_name(conf_node_name, node) && !(node.isReal() || node.isInt() || node.isString())) {
            return -1;
        }
        node >> scalar;
        return 0;
    }

};
} //seg_conf
} //utils
} //baidu

#endif
