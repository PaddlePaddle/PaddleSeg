#include <opencv2/opencv.hpp>

namespace {
class OrtSessionHandler {
 public:
  /**
   *  @param model_path path to onnx model
   *  @param gpu_idx index of the gpu, index < 0 means no gpu
   */
  OrtSessionHandler(const std::string &model_path, std::vector<std::vector<int>> &input_tensor_shapes,
                    int gpu_idx = -1);

  std::vector<float> preprocess(const cv::Mat &image, int target_height, int target_width,
                                const std::vector<float> &mean_val = {0.5, 0.5, 0.5},
                                const std::vector<float> &std_val = {0.5, 0.5, 0.5}) const;

 private:
  std::string _model_path;
  std::vector<std::vector<int>> _input_tensor_shapes;
  int _gpu_idx;
};

constexpr int BISENETV2_CITYSCAPES_IMAGE_HEIGHT = 1024;
constexpr int BISENETV2_CITYSCAPES_IMAGE_WIDTH = 1024;
static const std::vector<std::string> CITY_SCAPES_CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky",  "person",   "rider",    "car",  "truck", "bus",  "train",         "motorcycle",   "bicycle"};

static const std::vector<std::array<int, 3>> CITY_SCAPES_COLOR_CHART = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70},    {102, 102, 156}, {190, 153, 153}, {153, 153, 153}, {250, 170, 30},
    {220, 220, 0},  {107, 142, 35}, {152, 251, 152}, {70, 130, 180},  {220, 20, 60},   {255, 0, 0},     {0, 0, 142},
    {0, 0, 70},     {0, 60, 100},   {0, 80, 100},    {0, 0, 230},     {119, 11, 32}};
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: [app] [/path/to/image] [path/to/onnx/model] [gpu/idx]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string image_path = argv[1];
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cerr << "failed to load " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  const std::string onnx_model_path = argv[2];
  const int gpu_idx = std::atoi(argv[3]);

  std::vector<std::vector<int>> input_tensor_shapes{
      {1, 3, BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH}};
  OrtSessionHandler ort_session_handler(onnx_model_path, input_tensor_shapes, gpu_idx);
  auto input_data =
      ort_session_handler.preprocess(image, BISENETV2_CITYSCAPES_IMAGE_WIDTH, BISENETV2_CITYSCAPES_IMAGE_WIDTH);

  return EXIT_SUCCESS;
}

namespace {
OrtSessionHandler::OrtSessionHandler(const std::string &model_path, std::vector<std::vector<int>> &input_tensor_shapes,
                                     int gpu_idx)
    : _model_path(model_path), _input_tensor_shapes(input_tensor_shapes), _gpu_idx(gpu_idx) {}

std::vector<float> OrtSessionHandler::preprocess(const cv::Mat &image, int target_height, int target_width,
                                                 const std::vector<float> &mean_val,
                                                 const std::vector<float> &std_val) const {
  if (image.empty() || image.channels() != 3) {
    throw std::runtime_error("invalid image");
  }

  if (target_height * target_width == 0) {
    throw std::runtime_error("invalid dimension");
  }

  cv::Mat processed = image.clone();

  if (image.rows != target_height || image.cols != target_width) {
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
  }
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
  std::vector<float> data(3 * target_height * target_width);

  for (int i = 0; i < target_height; ++i) {
    for (int j = 0; j < target_width; ++j) {
      for (int c = 0; c < 3; ++c) {
        data[c * target_height * target_width + i * target_width + j] =
            (image.data[i * target_width * 3 + j * 3 + c] / 255.0 - mean_val[c]) / std_val[c];
      }
    }
  }

  return data;
}
}  // namespace
