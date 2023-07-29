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

# This implementation refers to: https://github.com/facebookresearch/segment-anything

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import time
import cv2  # type: ignore
import argparse
import numpy as np  # type: ignore
import paddle

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from paddleseg.utils.visualize import get_pseudo_color_map, get_color_map_list

ID_PHOTO_IMAGE_DEMO = "examples/cityscapes_demo.png"
CACHE_DIR = ".temp"

model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams",
    'vit_t':
    "https://paddleseg.bj.bcebos.com/dygraph/paddlesegAnything/vit_t/model.pdparam"
}

parser = argparse.ArgumentParser(description=(
    "Runs automatic mask generation on an input image or directory of images, "
    "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
    "as well as pycocotools if saving in RLE format."))

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_l",
    required=True,
    help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b', 'vit_t']", )

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."), )

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.", )

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.", )

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.", )

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."), )

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.", )

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."), )


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def delete_result():
    """clear old result in `.temp`"""
    results = sorted(os.listdir(CACHE_DIR))
    for res in results:
        if int(time.time()) - int(os.path.splitext(res)[0]) > 10000:
            os.remove(os.path.join(CACHE_DIR, res))


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

    # im = cv2.imread(image)
    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result


def gradio_display(generator):
    import gradio as gr

    def clear_image_all():
        delete_result()
        return None, None, None, None

    def get_id_photo_output(img):
        """
        Get the special size and background photo.

        Args:
            img(numpy:ndarray): The image array.
            size(str): The size user specified.
            bg(str): The background color user specified.
            download_size(str): The size for image saving.

        """
        predictor = generator
        masks = predictor.generate(img)
        pred_result, pseudo_map = masks2pseudomap(masks)  # PIL Image
        added_pseudo_map = visualize(
            img, pred_result, color_map=get_color_map_list(256))
        res_download = download(pseudo_map)

        return pseudo_map, added_pseudo_map, res_download

    with gr.Blocks() as demo:
        gr.Markdown("""# Segment Anything (PaddleSeg) """)
        with gr.Tab("InputImage"):
            image_in = gr.Image(value=ID_PHOTO_IMAGE_DEMO, label="Input image")

            with gr.Row():
                image_clear_btn = gr.Button("Clear")
                image_submit_btn = gr.Button("Submit")

            with gr.Row():
                img_out1 = gr.Image(
                    label="Output image", interactive=False).style(height=300)
                img_out2 = gr.Image(
                    label="Output image with mask",
                    interactive=False).style(height=300)
            downloaded_img = gr.File(label='Image download').style(height=50)

        image_clear_btn.click(
            fn=clear_image_all,
            inputs=None,
            outputs=[image_in, img_out1, img_out2, downloaded_img])

        image_submit_btn.click(
            fn=get_id_photo_output,
            inputs=[image_in, ],
            outputs=[img_out1, img_out2, downloaded_img])

        gr.Markdown(
            """<font color=Gray>Tips: You can try segment the default image OR upload any images you want to segment by click on the clear button first.</font>"""
        )

        gr.Markdown(
            """<font color=Gray>This is Segment Anything build with PaddlePaddle. 
            We refer to the [SAM](https://github.com/facebookresearch/segment-anything) for code strucure and model architecture.
            If you have any question or feature request, welcome to raise issues on [GitHub](https://github.com/PaddlePaddle/PaddleSeg/issues). </font>"""
        )

        gr.Button.style(1)

    demo.launch(server_name="0.0.0.0", server_port=8017, share=True)


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    sam = sam_model_registry[args.model_type](
        checkpoint=model_link[args.model_type])
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(
        sam, output_mode=output_mode, **amg_kwargs)

    gradio_display(generator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
