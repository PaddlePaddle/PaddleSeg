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

#include <string>

#include <google/protobuf/text_format.h>
#include "predictor/framework/memory.h"
#include "json2pb/pb_to_json.h"
#include "seg-serving/op/write_json_op.h"

namespace baidu {
namespace paddle_serving {
namespace predictor {

using json2pb::ProtoMessageToJson;
using baidu::paddle_serving::image_segmentation::ImageSegResponse;
using baidu::paddle_serving::image_segmentation::ResponseItem;
using baidu::paddle_serving::image_segmentation::Response;

int WriteJsonOp::inference() {

  const ImageSegResponse* seg_out =
      get_depend_argument<ImageSegResponse>("image_seg_op");

  if (!seg_out) {
    LOG(ERROR) << "Failed mutable depended argument, op:"
               << "image_seg_op";
    return -1;
  }

  Response* res = mutable_data<Response>();
  if (!res) {
    LOG(ERROR) << "Failed mutable output response in op:"
               << "WriteJsonOp";
    return -1;
  }

  // transfer seg output message into json format
  std::string err_string;
  uint32_t batch_size = seg_out->item_size();
  LOG(INFO) << "batch_size = " << batch_size;
//  LOG(INFO) << seg_out->ShortDebugString();
  for (uint32_t si = 0; si < batch_size; si++) {
    ResponseItem* ins = res->add_prediction();
    //LOG(INFO) << "Original image width = " << seg_out->width(si) << ", height = " << seg_out->height(si);
    if (!ins) {
      LOG(ERROR) << "Failed add one prediction ins";
      return -1;
    }
    std::string* text = ins->mutable_info();
    LOG(INFO) << seg_out->item(si).ShortDebugString();
    if (!ProtoMessageToJson(seg_out->item(si), text, &err_string)) {
      LOG(ERROR) << "Failed convert message["
                 << seg_out->item(si).ShortDebugString()
                 << "], err: " << err_string;
      return -1;
    }
  }
  return 0;
}

DEFINE_OP(WriteJsonOp);

}  // namespace predictor
}  // namespace paddle_serving
}  // namespace baidu
