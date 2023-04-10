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
from typing import List, Tuple, Type
from common import LayerNorm2d


class MaskDecoder(paddle.nn.Layer):
    def __init__(self,
                 *,
                 transformer_dim: int,
                 transformer: paddle.nn.Layer,
                 num_multimask_outputs: int=3,
                 activation: Type[paddle.nn.Layer]=paddle.nn.GELU,
                 iou_head_depth: int=3,
                 iou_head_hidden_dim: int=256) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = paddle.nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = paddle.nn.Embedding(self.num_mask_tokens,
                                               transformer_dim)
        self.output_upscaling = paddle.nn.Sequential(
            paddle.nn.Conv2DTranspose(
                in_channels=transformer_dim,
                out_channels=transformer_dim // 4,
                kernel_size=2,
                stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            paddle.nn.Conv2DTranspose(
                in_channels=transformer_dim // 4,
                out_channels=transformer_dim // 8,
                kernel_size=2,
                stride=2),
            activation())
        self.output_hypernetworks_mlps = paddle.nn.LayerList(sublayers=[
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(self,
                image_embeddings: paddle.Tensor,
                image_pe: paddle.Tensor,
                sparse_prompt_embeddings: paddle.Tensor,
                dense_prompt_embeddings: paddle.Tensor,
                multimask_output: bool) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(self,
                      image_embeddings: paddle.Tensor,
                      image_pe: paddle.Tensor,
                      sparse_prompt_embeddings: paddle.Tensor,
                      dense_prompt_embeddings: paddle.Tensor) -> Tuple[
                          paddle.Tensor, paddle.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        output_tokens = paddle.concat(
            x=[self.iou_token.weight, self.mask_tokens.weight], axis=0)
        output_tokens = output_tokens.unsqueeze(axis=0).expand(
            shape=[sparse_prompt_embeddings.shape[0], -1, -1])
        tokens = paddle.concat(
            x=(output_tokens, sparse_prompt_embeddings), axis=1)
        src = paddle.repeat_interleave(
            image_embeddings, tokens.shape[0], axis=0)
        src = src + dense_prompt_embeddings
        pos_src = paddle.repeat_interleave(image_pe, tokens.shape[0], axis=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, (0), :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        x = src
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        src = x.transpose(perm=perm_0).reshape([b, c, h, w])
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[paddle.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, (i), :]))
        hyper_in = paddle.stack(x=hyper_in_list, axis=1)
        b, c, h, w = upscaled_embedding.shape
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        masks = (hyper_in @upscaled_embedding.reshape([b, c, h * w])).reshape(
            [b, -1, h, w])
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class MLP(paddle.nn.Layer):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool=False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(sublayers=(paddle.nn.Linear(
            in_features=n,
            out_features=k) for n, k in zip([input_dim] + h, h + [output_dim])))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = paddle.nn.functional.relu(
                x=layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = paddle.nn.functional.sigmoid(x=x)
        return x
