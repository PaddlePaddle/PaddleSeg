# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import time
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle
import paddle.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.modeling.clip_paddle import build_clip_model, _transform
from segment_anything.utils.sample_tokenizer import tokenize
from paddleseg.utils.visualize import get_pseudo_color_map, get_color_map_list

ID_PHOTO_IMAGE_DEMO = "./examples/cityscapes_demo.png"
CACHE_DIR = ".temp"
model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams",
    'vit_t':
    "https://paddleseg.bj.bcebos.com/dygraph/paddlesegAnything/vit_t/model.pdparam",
    'clip_b_32':
    "https://bj.bcebos.com/paddleseg/dygraph/clip/vit_b_32_pretrain/clip_vit_b_32.pdparams"
}

parser = argparse.ArgumentParser(description=(
    "Runs automatic mask generation on an input image or directory of images, "
    "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
    "as well as pycocotools if saving in RLE format."))

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_h",
    required=True,
    help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b', 'vit_t']", )


def download(img):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    while True:
        name = str(int(time.time()))
        tmp_name = os.path.join(CACHE_DIR, name + '.jpg')
        if not os.path.exists(tmp_name):
            break
        else:
            time.sleep(1)
    img.save(tmp_name, 'png')
    return tmp_name


def segment_image(image, segment_mask):
    image_array = np.array(image)
    gray_image = Image.new("RGB", image.size, (128, 128, 128))
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segment_mask] = image_array[segment_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    transparency = np.zeros_like(segment_mask, dtype=np.uint8)
    transparency[segment_mask] = 255
    transparency_image = Image.fromarray(transparency, mode='L')
    gray_image.paste(segmented_image, mask=transparency_image)
    return gray_image


def image_text_match(cropped_objects, text_query):
    transformed_images = [transform(image) for image in cropped_objects]
    tokenized_text = tokenize([text_query])
    batch_images = paddle.stack(transformed_images)
    image_features = model.encode_image(batch_images)
    print("encode_image done!")
    text_features = model.encode_text(tokenized_text)
    print("encode_text done!")
    image_features /= image_features.norm(axis=-1, keepdim=True)
    text_features /= text_features.norm(axis=-1, keepdim=True)
    if len(text_features.shape) == 3:
        text_features = text_features.squeeze(0)
    probs = 100. * image_features @text_features.T
    return F.softmax(probs[:, 0], axis=0)


def masks2pseudomap(masks):
    result = np.ones(masks[0]["segmentation"].shape, dtype=np.uint8) * 255
    for i, mask_data in enumerate(masks):
        result[mask_data["segmentation"] == 1] = i + 1
    pred_result = result
    result = get_pseudo_color_map(result)
    return pred_result, result


def visualize(image, result, color_map, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result


def get_id_photo_output(image, text):
    """
    Get the special size and background photo.

    Args:
        img(numpy:ndarray): The image array.
        size(str): The size user specified.
        bg(str): The background color user specified.
        download_size(str): The size for image saving.

    """
    image_ori = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    pred_result, pseudo_map = masks2pseudomap(masks)  # PIL Image
    added_pseudo_map = visualize(
        image, pred_result, color_map=get_color_map_list(256))
    cropped_objects = []
    image_pil = Image.fromarray(image)
    for mask in masks:
        bbox = [
            mask["bbox"][0], mask["bbox"][1], mask["bbox"][0] + mask["bbox"][2],
            mask["bbox"][1] + mask["bbox"][3]
        ]
        cropped_objects.append(
            segment_image(image_pil, mask["segmentation"]).crop(bbox))

    scores = image_text_match(cropped_objects, str(text))
    text_matching_masks = []
    for idx, score in enumerate(scores):
        if score < 0.05:
            continue
        text_matching_mask = Image.fromarray(
            masks[idx]["segmentation"].astype('uint8') * 255)
        text_matching_masks.append(text_matching_mask)

    image_pil_ori = Image.fromarray(image_ori)
    alpha_image = Image.new('RGBA', image_pil_ori.size, (0, 0, 0, 0))
    alpha_color = (255, 0, 0, 180)

    draw = ImageDraw.Draw(alpha_image)
    for text_matching_mask in text_matching_masks:
        draw.bitmap((0, 0), text_matching_mask, fill=alpha_color)

    result_image = Image.alpha_composite(
        image_pil_ori.convert('RGBA'), alpha_image)
    res_download = download(result_image)
    return result_image, added_pseudo_map, res_download


def gradio_display():
    import gradio as gr
    examples_sam = [["./examples/cityscapes_demo.png", "a photo of car"],
                    ["examples/dog.jpg", "dog"],
                    ["examples/zixingche.jpeg", "kid"]]

    demo_mask_sam = gr.Interface(
        fn=get_id_photo_output,
        inputs=[
            gr.Image(
                value=ID_PHOTO_IMAGE_DEMO,
                label="Input image").style(height=400), gr.inputs.Textbox(
                    lines=3,
                    placeholder=None,
                    default="a photo of car",
                    label='🔥 Input text prompt 🔥',
                    optional=False)
        ],
        outputs=[
            gr.Image(
                label="Output based on text",
                interactive=False).style(height=300), gr.Image(
                    label="Output mask", interactive=False).style(height=300)
        ],
        examples=examples_sam,
        description="<p> \
                        <strong>SAM+CLIP:  Text prompt for segmentation. </strong> <br>\
                        Choose an example below; Or, upload by yourself: <br>\
                        1. Upload images to be tested to 'input image'. 2. Input a text prompt to 'input text prompt' and click 'submit'</strong>.  <br>\
                        </p>",
        cache_examples=False,
        allow_flagging="never", )

    demo = gr.TabbedInterface(
        [demo_mask_sam, ], ['SAM+CLIP(Text to Segment)'],
        title=" 🔥 Text to Segment Anything with PaddleSeg 🔥")
    demo.launch(
        server_name="0.0.0.0", enable_queue=False, server_port=8078, share=True)


args = parser.parse_args()
print("Loading model...")

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")

sam = sam_model_registry[args.model_type](
    checkpoint=model_link[args.model_type])
mask_generator = SamAutomaticMaskGenerator(sam)

model, transform = build_clip_model(model_link["clip_b_32"])
gradio_display()
