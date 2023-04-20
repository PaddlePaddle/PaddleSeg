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

# This implementation refers to: https://github.com/openai/CLIP

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import paddle
from paddle.nn.initializer import Constant
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

zeros_ = Constant(value=0.)


class QuickGELU(paddle.nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=1.702 * x)


class ResidualAttentionBlock(paddle.nn.Layer):
    def __init__(self, d_model: int, n_head: int,
                 attn_mask: paddle.Tensor=None):
        super().__init__()
        self.attn = paddle.nn.MultiHeadAttention(
            d_model,
            n_head,
            need_weights=False, )
        self.ln_1 = paddle.nn.LayerNorm(d_model)
        self.mlp = paddle.nn.Sequential(*[('c_fc', paddle.nn.Linear(
            in_features=d_model, out_features=d_model *
            4)), ('gelu', QuickGELU()), ('c_proj', paddle.nn.Linear(
                in_features=d_model * 4, out_features=d_model))])
        self.ln_2 = paddle.nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: paddle.Tensor):
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        self.attn_mask = self.attn_mask.astype(
            x.dtype) if self.attn_mask is not None else None
        x = x.transpose([1, 0, 2])
        x = self.attn(x, x, x, attn_mask=self.attn_mask)
        return x.transpose([1, 0, 2])

    def forward(self, x: paddle.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(paddle.nn.Layer):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: paddle.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = paddle.nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: paddle.Tensor):
        return self.resblocks(x)


class VisionTransformer(paddle.nn.Layer):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=False)
        scale = width**-0.5
        self.class_embedding = self.create_parameter(
            shape=(width, ), default_initializer=zeros_)
        self.add_parameter("class_embedding", self.class_embedding)

        self.positional_embedding = self.create_parameter(
            shape=((input_resolution // patch_size)**2 + 1, width),
            default_initializer=zeros_)
        self.add_parameter("positional_embedding", self.positional_embedding)
        self.ln_pre = paddle.nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = paddle.nn.LayerNorm(width)
        self.proj = self.create_parameter(
            shape=(width, output_dim), default_initializer=zeros_)
        self.add_parameter("proj", self.proj)

    def forward(self, x: paddle.Tensor):

        x = self.conv1(x)
        x = x.flatten(2).transpose((0, 2, 1))

        x = paddle.concat(
            [
                self.class_embedding.astype(x.dtype) + paddle.zeros(
                    shape=[x.shape[0], 1, x.shape[-1]], dtype=x.dtype), x
            ],
            axis=1)

        x = x + self.positional_embedding.astype(x.dtype)
        x = self.ln_pre(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @self.proj
        return x


class CLIP(paddle.nn.Layer):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = paddle.nn.Embedding(vocab_size,
                                                   transformer_width)

        self.positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=zeros_)
        self.add_parameter("positional_embedding", self.positional_embedding)

        self.ln_final = paddle.nn.LayerNorm(transformer_width)

        self.text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim), default_initializer=zeros_)
        self.add_parameter("text_projection", self.text_projection)

    def build_attention_mask(self):
        mask = paddle.empty(shape=[self.context_length, self.context_length])
        mask.fill_(value=float('-inf'))
        mask = paddle.tensor.triu(mask, diagonal=1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(self.dtype)
        x = x + self.positional_embedding.astype(self.dtype)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.ln_final(x).astype(self.dtype)
        x = x[paddle.arange(start=x.shape[0]), text.argmax(
            axis=-1)] @self.text_projection
        return x[None, :]

    def forward(self, image, text):
        text_features = self.encode_text(text)[None, :]
        image_features = self.encode_image(image)

        image_features = image_features / image_features.norm(
            axis=1, keepdim=True)
        text_features = text_features / text_features.norm(axis=1, keepdim=True)
        # cosine similarity as logits
        logits_per_image = image_features @text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def convert_weights(model: paddle.nn.Layer):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(param):
        if isinstance(param,
                      (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Linear)):
            param.weight.data = param.weight.astype('float16')
            if param.bias is not None:
                param.bias.data = param.bias.astype('float16')
        if isinstance(param, paddle.nn.MultiHeadAttention):
            for attr in [
                    * [f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor_l = getattr(param, attr)
                if tensor_l is not None:
                    tensor_l.data = tensor_l.astype('float16')
        for name in ['text_projection', 'proj']:
            if hasattr(param, name):
                attr = getattr(param, name)
                if attr is not None:
                    attr.data = attr.astype('float16')

    model.apply(fn=_convert_weights_to_fp16)


def load_pretrain_clip(pretrained_model):
    from urllib.parse import urlparse
    from paddleseg.utils import download_pretrained_model
    if urlparse(pretrained_model).netloc:
        pretrained_model = download_pretrained_model(pretrained_model)
    state = paddle.load(pretrained_model)
    return state


def build_clip_model(pretrained_model):
    state_dict = load_pretrain_clip(pretrained_model)
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.q_proj.weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] -
                           1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        counts: list = [
            len(
                set(
                    k.split('.')[2] for k in state_dict
                    if k.startswith(f'visual.layer{b}')))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding']
                              .shape[0] - 1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            'visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    transformer_layers = len(
        set(
            k.split('.')[2] for k in state_dict
            if k.startswith('transformer.resblocks')))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    #convert_weights(model)
    model.eval()
    model.set_state_dict(state_dict=state_dict)
    return model, _transform(model.visual.input_resolution)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):

    return Compose([
        Resize(n_px),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
