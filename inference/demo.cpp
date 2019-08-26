#include <glog/logging.h>
#include <utils/utils.h>
#include <predictor/seg_predictor.h>

DEFINE_string(conf, "", "Configuration File Path");
DEFINE_string(input_dir, "", "Directory of Input Images");

int main(int argc, char** argv) {
    // 0. parse args
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_conf.empty() || FLAGS_input_dir.empty()) {
        std::cout << "Usage: ./predictor --conf=/config/path/to/your/model --input_dir=/directory/of/your/input/images";
        return -1;
    }
    // 1. create a predictor and init it with conf
    PaddleSolution::Predictor predictor;
    if (predictor.init(FLAGS_conf) != 0) {
        LOG(FATAL) << "Fail to init predictor";
        return -1;
    }

    // 2. get all the images with extension '.jpeg' at input_dir
    auto imgs = PaddleSolution::utils::get_directory_images(FLAGS_input_dir, ".jpeg|.jpg");

    // 3. predict
    predictor.predict(imgs);
    return 0;
}
