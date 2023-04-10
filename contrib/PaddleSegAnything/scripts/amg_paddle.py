# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import time
import cv2  # type: ignore
import json
import argparse
import numpy as  np  # type: ignore
from typing import Any, Dict, List
from collections import OrderedDict
import matplotlib.pyplot as plt
import paddle

from paddleseg_anything import SamAutomaticMaskGenerator, sam_model_registry
from paddleseg.utils.visualize import get_pseudo_color_map


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

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
    help="How many input points to process simultaneously in one batch.",
)

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
    help="Exclude masks with a stability score lower than this threshold.",
)

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
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

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
    help="Larger numbers mean image crops will overlap more.",
)

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
        "in pixels are removed by postprocessing."
    ),
)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


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


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    paddle.set_device("gpu")
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]


    for t in targets:
        print(f"Processing '{t}'...")
        time1 = time.time()
        image = cv2.imread(t)
        if image is None:
            raise FileNotFoundError('{} is not found.'.format(t))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks, save_base) # repalce with pseudo mask
        elif output_mode == "pseudo_color_mask":
            result = np.ones(masks[0]["segmentation"].shape, dtype=np.uint8) * 255
            for i, mask_data in enumerate(masks):
                result[mask_data["segmentation"] == 1] = i+1
            result = get_pseudo_color_map(result)
            basename = f'{base}_mask.png'
            result.save(os.path.join(args.output, basename))
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done with {} s!".format(time.time()-time1))

    #gradio_display(generator)


if __name__ == "__main__":
    SIZES = OrderedDict({
        "1 inch": {
            'physics': (25, 35),
            'pixels': (295, 413)
        },
        "1 inch smaller": {
            'physics': (22, 32),
            'pixels': (260, 378)
        },
        "1 inch larger": {
            'physics': (33, 48),
            'pixels': (390, 567)
        },
        "2 inches": {
            'physics': (35, 49),
            'pixels': (413, 579)
        },
        "2 inches smaller": {
            'physics': (35, 45),
            'pixels': (413, 531)
        },
        "2 inches larger": {
            'physics': (35, 53),
            'pixels': (413, 626)
        },
        "3 inches": {
            'physics': (55, 84),
            'pixels': (649, 991)
        },
        "4 inches": {
            'physics': (76, 102),
            'pixels': (898, 1205)
        },
        "5 inches": {
            'physics': (89, 127),
            'pixels': (1050, 1500)
        }
    })

    # jpg compress ratio
    SAVE_SIZE = {'Small': 50, 'Middle': 75, 'Large': 95}

    CACHE_DIR = ".temp"


    def delete_result():
        """clear old result in `.temp`"""
        results = sorted(os.listdir(CACHE_DIR))
        for res in results:
            if int(time.time()) - int(os.path.splitext(res)[0]) > 10000:
                os.remove(os.path.join(CACHE_DIR, res))

    def adjust_size(img, size_index):
        key = list(SIZES.keys())[size_index]
        w_o, h_o = SIZES[key]['pixels']

        # scale
        img = np.array(img)
        h_ori, w_ori = img.shape[:2]
        scale = max(w_o / w_ori, h_o / h_ori)
        if scale > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        img_scale = cv2.resize(
            img, dsize=None, fx=scale, fy=scale, interpolation=interpolation)

        # crop
        h_scale, w_scale = img_scale.shape[:2]
        h_cen = h_scale // 2
        w_cen = w_scale // 2
        h_start = max(0, h_cen - h_o // 2)
        h_end = min(h_scale, h_start + h_o)
        w_start = max(0, w_cen - w_o // 2)
        w_end = min(w_scale, w_start + w_o)
        img_c = img_scale[h_start:h_end, w_start:w_end]

        return img_c


    def download(img, size):
        q = SAVE_SIZE[size]
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
            result[mask_data["segmentation"] == 1] = i+1
        result = get_pseudo_color_map(result)

        return result

    def gradio_display(generator):
        import gradio as gr

        ID_PHOTO_IMAGE_DEMO = "cityscapes_demo.png"

        def clear_image_all():
            delete_result()
            return None, None, 'Large', None
        
        def get_id_photo_output(img, download_size):
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
            pseudo_map = masks2pseudomap(masks) # PIL Image
            res_download = download(pseudo_map, download_size)
            
            return pseudo_map, res_download

        with gr.Blocks() as demo:
            gr.Markdown("""# Paddleseg Everything Demo""")
            gr.Markdown("""<font color=Gray>Tips: Please upload any photos you want to segment.</font>""")
            with gr.Tab("InputImage"):
                image_in = gr.Image(value=ID_PHOTO_IMAGE_DEMO, label="Input image")
                with gr.Row():
                    image_download_size = gr.Radio(
                        ["Small", "Middle", "Large"],
                        label="Download file size (affects image quality)",
                        value='Large',
                        interactive=True)

                with gr.Row():
                    image_clear_btn = gr.Button("Clear")
                    image_submit_btn = gr.Button("Submit")

                img_out = gr.Image(label="Output image", interactive=False).style(height=300)
                downloaded_img = gr.File(label='Image download').style(height=50)
        
            image_clear_btn.click(
                fn=clear_image_all,
                inputs=None,
                outputs=[image_in, img_out,
                        image_download_size, downloaded_img])

            image_submit_btn.click(
                fn=get_id_photo_output,
                inputs=[image_in, image_download_size],
                outputs=[img_out, downloaded_img])

            gr.Markdown(
                """<font color=Gray>This application is supported by [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg). 
                If you have any question or feature request, 
                welcome to raise issues on [GitHub](https://github.com/PaddlePaddle/PaddleSeg/issues). 
                BTW, a star is a great encouragement for us, thanks!  ^_^</font>"""
            )

            gr.Button.style(1)

        demo.launch(server_name="0.0.0.0", server_port=8017, share=True)

    args = parser.parse_args()
    main(args)