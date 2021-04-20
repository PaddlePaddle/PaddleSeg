#include "seg_conf.h"
DEFINE_string(seg_conf_file, "conf/seg_conf.yaml", "seg configuration filename");

namespace baidu{
namespace utils{
namespace seg_conf{

SegConf::SegConf(const std::string &configuration_filename) {
    std::cout << "filename: " << configuration_filename << std::endl;
    try{
        if(!_seg_conf_file.open(configuration_filename, cv::FileStorage::READ)){
                std::cout << "Configuration file open error!" << std::endl;
        }
    } catch(...){
        std::cout << "error" << std::endl;
    }

}

SegConf::~SegConf(){
    _seg_conf_file.release();
}

bool SegConf::get_item_by_name(const std::string &conf_node_name, cv::FileNode &return_file_node) const{
    return_file_node = _seg_conf_file[conf_node_name];
    if(return_file_node.isNone()) {
        std::cout << "You haven't configure this item" << std::endl;
        return false;
    }
    return true;
}

int SegConf::get_mean_vector(std::vector<double> &mean_vec) const {
    return get_array_from_file_node("MEAN", mean_vec);
}

int SegConf::get_std_vector(std::vector<double> &std_vec) const{
    return get_array_from_file_node("STD", std_vec);
}

int SegConf::get_size_vector(std::vector<int> &size_vec) const{
    return get_array_from_file_node("SIZE", size_vec);
}

int SegConf::get_channels(int &channels) const{
    return get_scalar_from_file_node("CHANNELS", channels);
}

int SegConf::get_class_num(int &class_num) const {
    return get_scalar_from_file_node("CLASS_NUM", class_num);
}

int SegConf::get_model_name(std::string &name) const {
    return get_scalar_from_file_node("MODEL_NAME", name);
}

const SegConf* SegConf::instance() {
//lock
    static const SegConf s_seg_conf_instance(FLAGS_seg_conf_file);

    return &s_seg_conf_instance;
}

//SegConf SegConf::s_seg_conf_instance(FLAGS_seg_conf_file);

} //seg_conf
} //utils
} //baidu
