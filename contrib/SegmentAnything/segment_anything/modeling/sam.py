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

import paddle
from paddle import nn
from paddle.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .tiny_vit_sam import TinyViT


class Sam(nn.Layer):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: Union[ImageEncoderViT, TinyViT],
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float]=[123.675, 116.28, 103.53],
            pixel_std: List[float]=[58.395, 57.12, 57.375], ) -> None:
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

    @paddle.no_grad()
    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool, ) -> List[Dict[str, paddle.Tensor]]:
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
            if "point_coords" in image_record:
                points = (image_record["point_coords"],
                          image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None), )
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
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
