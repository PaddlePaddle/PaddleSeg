# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import paddle
import yaml
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Export Inference Model.')
    # parser.add_argument("--config", default='./config.yml', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        default="./model.pdparams",
        help='The path of trained weights for exporting inference model',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the exported inference model',
        type=str,
        default='./inference_model')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int,
        default=None)
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help="Select the op to be appended to the last of inference model, default: argmax."
        "In PaddleSeg, the output of trained model is logit (H*C*H*W). We can apply argmax and"
        "softmax op to the logit according the actual situation.")

    return parser.parse_args()


def main(args):

    import paddle
    from paddle import nn
    from paddle.nn import functional as F

    from typing import Any, Dict, List, Tuple

    from segment_anything.modeling.image_encoder import ImageEncoderViT
    from segment_anything.modeling.mask_decoder import MaskDecoder
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from functools import partial
    from paddleseg.utils import load_entire_model
    from segment_anything.modeling.transformer import TwoWayTransformer

    class Sam(nn.Layer):
        mask_threshold: float = 0.0
        image_format: str = "RGB"

        def __init__(
                self,
                image_encoder: ImageEncoderViT,
                prompt_encoder: PromptEncoder,
                mask_decoder: MaskDecoder,
                pixel_mean: List[float] = [123.675, 116.28, 103.53],
                pixel_std: List[float] = [58.395, 57.12, 57.375], ) -> None:
            """
            SAM predicts object masks from an image and input prompts.

            Arguments:
              image_encoder (ImageEncoderViT): The backbone used to encode the
                image into image embeddings that allow for efficient mask prediction.
              prompt_encoder (PromptEncoder): Encodes various types of input prompts.
              mask_decoder (MaskDecoder): Predicts masks from the image embeddings
                and encoded prompts.
              pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
              pixel_std (list(float)): Std values for normalizing pixels in the input image.
            """
            super().__init__()
            self.image_encoder = image_encoder
            self.prompt_encoder = prompt_encoder
            self.mask_decoder = mask_decoder
            self.register_buffer(
                "pixel_mean",
                paddle.to_tensor(pixel_mean).reshape([-1, 1, 1]),
                persistable=False)
            self.register_buffer(
                "pixel_std",
                paddle.to_tensor(pixel_std).reshape([-1, 1, 1]),
                persistable=False)

        @property
        def device(self) -> Any:
            if paddle.is_compiled_with_cuda():
                return 'gpu'
            else:
                return 'cpu'

        # @paddle.no_grad()
        def forward(
                self,
                batched_input: List[Dict[str, Any]],
                multimask_output=True, ) -> List[Dict[str, paddle.Tensor]]:
            """
            Predicts masks end-to-end from provided images and prompts.
            If prompts are not known in advance, using SamPredictor is
            recommended over calling the model directly.

            Arguments:
              batched_input (list(dict)): A list over input images, each a
                dictionary with the following keys. A prompt key can be
                excluded if it is not present.
                  'image': The image as a paddle tensor in 3xHxW format,
                    already transformed for input to the model.
                  'original_size': (tuple(int, int)) The original size of
                    the image before transformation, as (H, W).
                  'point_coords': (paddle.Tensor) Batched point prompts for
                    this image, with shape BxNx2. Already transformed to the
                    input frame of the model.
                  'point_labels': (paddle.Tensor) Batched labels for point prompts,
                    with shape BxN.
                  'boxes': (paddle.Tensor) Batched box inputs, with shape Bx4.
                    Already transformed to the input frame of the model.
                  'mask_inputs': (paddle.Tensor) Batched mask inputs to the model,
                    in the form Bx1xHxW.
              multimask_output (bool): Whether the model should predict multiple
                disambiguating masks, or return a single mask.

            Returns:
              (list(dict)): A list over input images, where each element is
                as dictionary with the following keys.
                  'masks': (paddle.Tensor) Batched binary mask predictions,
                    with shape BxCxHxW, where B is the number of input promts,
                    C is determiend by multimask_output, and (H, W) is the
                    original size of the image.
                  'iou_predictions': (paddle.Tensor) The model's predictions
                    of mask quality, in shape BxC.
                  'low_res_logits': (paddle.Tensor) Low resolution logits with
                    shape BxCxHxW, where H=W=256. Can be passed as mask input
                    to subsequent iterations of prediction.
            """
            input_images = paddle.stack(
                [self.preprocess(x["image"]) for x in batched_input], axis=0)
            image_embeddings = self.image_encoder(input_images)

            outputs = []
            for image_record, curr_embedding in zip(batched_input,
                                                    image_embeddings):
                # if "point_coords" in image_record:
                #     points = (image_record["point_coords"],
                #               image_record["point_labels"])
                # else:
                #     points = None

                # sparse_embeddings, dense_embeddings = self.prompt_encoder(
                #     points=points,
                #     boxes=image_record.get("boxes", None),
                #     masks=image_record.get("mask_inputs", None), )
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None, )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output, )
                masks = self.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"], )
                masks = masks > self.mask_threshold
                outputs.append({
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                })
            return outputs

        def postprocess_masks(
                self,
                masks: paddle.Tensor,
                input_size: Tuple[int, ...],
                original_size: Tuple[int, ...], ) -> paddle.Tensor:
            """
            Remove padding and upscale masks to the original image size.

            Arguments:
              masks (paddle.Tensor): Batched masks from the mask_decoder,
                in BxCxHxW format.
              input_size (tuple(int, int)): The size of the image input to the
                model, in (H, W) format. Used to remove padding.
              original_size (tuple(int, int)): The original size of the image
                before resizing for input to the model, in (H, W) format.

            Returns:
              (paddle.Tensor): Batched masks in BxCxHxW format, where (H, W)
                is given by original_size.
            """
            masks = F.interpolate(
                masks,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False, )
            masks = masks[..., :input_size[0], :input_size[1]]
            masks = F.interpolate(
                masks, original_size, mode="bilinear", align_corners=False)
            return masks

        def preprocess(self, x: paddle.Tensor) -> paddle.Tensor:
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            h, w = x.shape[-2:]
            padh = self.image_encoder.img_size - h
            padw = self.image_encoder.img_size - w
            # import pdb; pdb.set_trace()
            # x = F.pad(x, (0, padw, 0, padh))
            padding = paddle.full([4], 0, dtype='int32')
            padding[1] = padw
            padding[3] = padh
            x = F.pad(x, padding)
            return x.squeeze(0)


    # class WrappedModel(paddle.nn.Layer):
    #     def __init__(self, model, output_op):
    #         super().__init__()
    #         self.model = model
    #         self.output_op = output_op
    #         assert output_op in ['argmax', 'softmax'], \
    #             "output_op should in ['argmax', 'softmax']"
    #
    #     def forward(self, x, y):
    #         # import pdb; pdb.set_trace()
    #         outs = self.model(x, y)
    #         new_outs = []
    #         for out in outs:
    #             if self.output_op == 'argmax':
    #                 out = paddle.argmax(out, axis=1, dtype='int32')
    #             elif self.output_op == 'softmax':
    #                 out = paddle.nn.functional.softmax(out, axis=1)
    #             new_outs.append(out)
    #         return new_outs

    encoder_embed_dim = 1280
    encoder_depth = 32
    encoder_num_heads = 16
    encoder_global_attn_indexes = [7, 15, 23, 31]
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size


    # save model
    model = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(
                paddle.nn.LayerNorm, epsilon=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim, ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16, ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8, ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256, ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375], )

    if args.model_path is not None:
        state_dict = paddle.load(args.model_path)
        model.set_dict(state_dict)
        logger.info('Loaded trained params successfully.')
    # if args.output_op != 'none':
    #     model = WrappedModel(model, args.output_op)

    shape = [None, 3, None, None] if args.input_shape is None \
        else args.input_shape
    # input_spec = [[paddle.static.InputSpec(shape=[None], name='batched_input')],
                  # paddle.static.InputSpec(shape=[None], dtype='bool', name='multimask_output')]
    input_spec = [[{
                    'image': paddle.static.InputSpec(shape=[3, None, None], name='image'),
                    'original_size': paddle.static.InputSpec(shape=[None, None], name='original_size'),
                    # 'point_coords': paddle.static.InputSpec(shape=[None], name='point_coords'),
                    # 'point_labels': paddle.static.InputSpec(shape=[None], name='point_labels'),
                    # 'boxes': paddle.static.InputSpec(shape=[None, None], name='boxes'),
                    # 'mask_inputs': paddle.static.InputSpec(shape=[None], name='mask_inputs')
                    }],
                  # paddle.static.InputSpec(shape=[None], name='multimask_output')
        True
    ]

    # output_spec = [
    #     "masks": masks,
    #     "iou_predictions": iou_predictions,
    #     "low_res_logits": low_res_masks]

    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(model, os.path.join(args.save_dir, 'model'))

    # output_dtype = 'int32' if args.output_op == 'argmax' else 'float32'

    # TODO add test config
    deploy_info = {
        'Deploy': {
            'model': 'model.pdmodel',
            'params': 'model.pdiparams',
            # 'transforms': transforms,
            'input_shape': shape,
            # 'output_op': args.output_op,
            # 'output_dtype': output_dtype
        }
    }
    msg = '\n---------------Deploy Information---------------\n'
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        yaml.dump(deploy_info, file)

    logger.info(f'The inference model is saved in {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
