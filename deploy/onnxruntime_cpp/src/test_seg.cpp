#include "ort_session_handler.hpp"

namespace {
constexpr int BISENETV2_CITYSCAPES_IMAGE_HEIGHT = 1024;
constexpr int BISENETV2_CITYSCAPES_IMAGE_WIDTH = 1024;

static const std::vector<std::vector<uint8_t>> CITYSCAPES_COLORS = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70},    {102, 102, 156}, {190, 153, 153}, {153, 153, 153}, {250, 170, 30},
    {220, 220, 0},  {107, 142, 35}, {152, 251, 152}, {70, 130, 180},  {220, 20, 60},   {255, 0, 0},     {0, 0, 142},
    {0, 0, 70},     {0, 60, 100},   {0, 80, 100},    {0, 0, 230},     {119, 11, 32}};
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: [app] [/path/to/image] [path/to/onnx/model]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string image_path = argv[1];
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cerr << "failed to load " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  const std::string onnx_model_path = argv[2];

  std::vector<std::vector<int64_t>> input_tensor_shapes{
      {1, 3, BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH}};
  deploy::OrtSessionHandler ort_session_handler(onnx_model_path, input_tensor_shapes);
  std::vector<float> input_data =
      ort_session_handler.preprocess(image, BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH);

  // output data's type might change for each different model
  auto output_data = ort_session_handler.run<int64_t>({input_data});

  // postprocess
  // this might change for each different model
  cv::Mat segm(BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH, CV_8UC(3));
  for (int i = 0; i < BISENETV2_CITYSCAPES_IMAGE_HEIGHT; ++i) {
    cv::Vec3b *ptr_segm = segm.ptr<cv::Vec3b>(i);
    for (int j = 0; j < BISENETV2_CITYSCAPES_IMAGE_WIDTH; ++j) {
      const auto &color = CITYSCAPES_COLORS[output_data[0].first[i * BISENETV2_CITYSCAPES_IMAGE_WIDTH + j]];
      ptr_segm[j] = cv::Vec3b(color[0], color[1], color[2]);
    }
  }
  cv::resize(segm, segm, image.size(), 0, 0, cv::INTER_NEAREST);
  float blended_alpha = 0.4;
  segm = (1 - blended_alpha) * image + blended_alpha * segm;
  cv::imwrite("out_img.jpg", segm);

  return EXIT_SUCCESS;
}
