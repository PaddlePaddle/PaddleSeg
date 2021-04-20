// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "predictor/framework/infer.h"
#include "predictor/framework/memory.h"
#include "seg-serving/op/image_seg_op.h"
#include "seg-serving/op/reader_op.h"
#include "seg-serving/op/seg_conf.h"
namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::image_segmentation::ImageSegResItem;
using baidu::paddle_serving::image_segmentation::ImageSegResponse;
using baidu::paddle_serving::predictor::InferManager;
using baidu::utils::seg_conf::SegConf;
int ImageSegOp::inference() {
  const ReaderOutput* reader_out =
      get_depend_argument<ReaderOutput>("image_reader_op");
  if (!reader_out) {
    LOG(ERROR) << "Failed mutable depended argument, op:"
               << "reader_op";
    return -1;
  }
  const TensorVector* in = &reader_out->tensors;

  const std::vector<int> *width_vec = &reader_out->width_vec;
  const std::vector<int> *height_vec = &reader_out->height_vec;
  //debug
  for(int i = 0; i < width_vec->size(); ++i){
      LOG(INFO) << "width = " << (*width_vec)[i] << ", height = " << (*height_vec)[i];
  }


  TensorVector* out = butil::get_object<TensorVector>();
  if (!out) {
    LOG(ERROR) << "Failed get tls output object failed";
    return -1;
  }

  if (in->size() != 1) {
    LOG(ERROR) << "Samples should have been packed into a single tensor";
    return -1;
  }

  int batch_size = in->at(0).shape[0];

  static const SegConf *sc_ptr = SegConf::instance();
  // call paddle fluid model for inferencing
  std::string model_name;
  sc_ptr->get_model_name(model_name);
  LOG(INFO) << "model name = " << model_name;
  int ret;
  if ((ret = InferManager::instance().infer(
          model_name.c_str(), in, out, batch_size))) {
    LOG(ERROR) << "Failed do infer in fluid model: "
               << model_name;
    return -1;
  }
  LOG(INFO) << "ret = " << ret;
  if (out->size() != in->size()) {
    LOG(ERROR) << "Output size not eq input size: " << in->size()
               << out->size();
    return -1;
  }

  // copy output tensor into response
  ImageSegResponse* res = mutable_data<ImageSegResponse>();
  const paddle::PaddleTensor& out_tensor = (*out)[0];

  int sample_size = out_tensor.shape[0];

  uint32_t total_size = 1;
  for (int i = 0; i < out_tensor.shape.size(); ++i) {
      total_size *= out_tensor.shape[i];
  }
  LOG(INFO) << "total_size = " << total_size;
  uint32_t item_size = total_size / sample_size;
  for (uint32_t si = 0; si < sample_size; si++) {
    ImageSegResItem* ins = res->add_item();
//    res->add_width((*width_vec)[si]);
//    res->add_height((*height_vec)[si]);
    if (!ins) {
      LOG(ERROR) << "Failed append new out tensor";
      return -1;
    }

    // assign output data
    float* data = reinterpret_cast<float*>(out_tensor.data.data() +
                                           si * sizeof(float) * item_size);
    std::vector<int> size_vec;
    sc_ptr->get_size_vector(size_vec);
    int width = size_vec[0];
    int height = size_vec[1];
    int class_num;
    sc_ptr->get_class_num(class_num);
    LOG(INFO) << "width = " << width << ", height = " << height << ", class_num = " << class_num;
    uint32_t out_size = width * height;
    mask_raw.clear();
    mask_raw.resize(out_size);
    for (uint32_t di = 0; di < out_size; ++di) {
        float max_value = -1;
        int label = 0;
        for (int j = 0; j < class_num; ++j) {
            int index = di + j * out_size;
            if (index >= class_num * width * height) {
              break;
            }
            float value = data[index];
            if (value > max_value){
                max_value = value;
                label = j;
            }
        }
        if (label == 0) max_value = 0;
        mask_raw[di] = label;
    }

    //cv::Mat mask_mat = cv::Mat(height, width, CV_32FC1);
    cv::Mat mask_mat = cv::Mat(height, width, CV_8UC1);
    //scoremap
   // mask_mat.data = reinterpret_cast<uchar *>(data + out_size);
    //mask_mat.data = mask_raw.data();
    std::vector<uchar> temp_mat(out_size, 0);
    for(int i = 0; i < out_size; ++i){
        temp_mat[i] = 255 * data[i + out_size];
    }
    mask_mat.data = temp_mat.data();

    cv::Mat mask_temp_mat((*height_vec)[si], (*width_vec)[si], mask_mat.type());
    //Size(cols, rows)
    cv::resize(mask_mat, mask_temp_mat, mask_temp_mat.size());
//debug
    //for(int i = 0; i < (*height_vec)[si]; ++i){
    //    for(int j = 0; j < (*width_vec)[si]; ++j) {
    //      std::cout << mask_temp_mat.at<float>(i, j) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    std::vector<uchar> mat_buff;
    cv::imencode(".png", mask_temp_mat, mat_buff);
    ins->set_mask(reinterpret_cast<char *>(mat_buff.data()), mat_buff.size());
  }

  // release out tensor object resource
  size_t out_size = out->size();
  for (size_t oi = 0; oi < out_size; ++oi) {
    (*out)[oi].shape.clear();
  }
  out->clear();
  butil::return_object<TensorVector>(out);
  return 0;
}

DEFINE_OP(ImageSegOp);

}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
