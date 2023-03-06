# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
#
# This implementation refers to: https://github.com/facebookresearch/MaskFormer/tree/main/mask_former/modeling

import math
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils
from paddleseg.core.train import check_logits_losses


@manager.MODELS.add_component
class MaskFormer(nn.Layer):
    """
    The MaskFormer model implement on PaddlePaddle.
    
    The original article please refer to :
    Cheng, Bowen, Alex Schwing, and Alexander Kirillov. "Per-pixel classification is not all you need for semantic segmentation." Advances in Neural Information Processing Systems 34 (2021): 17864-17875.
    (https://github.com/facebookresearch/MaskFormer)

    Args:
        num_classes(int): The number of classes that you want the model to classify.
        backbone(nn.Layer): The backbone module defined in the paddleseg backbones.
        sem_seg_postprocess_before_inference(bool): If True, do result postprocess before inference. 
        pretrained(str): The path to the pretrained model of MaskFormer.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 sem_seg_postprocess_before_inference=False,
                 pretrained=None):
        super(MaskFormer, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.seghead = MaskFormerHead(backbone.output_shape(), num_classes)
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls)[..., :-1]
        mask_pred = F.sigmoid(mask_pred)
        semseg = paddle.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.seghead(features)

        if self.training:
            return [outputs]
        else:
            mask_cls_results = outputs["pred_logits"]  # [2, 100, 151]
            mask_pred_results = outputs["pred_masks"]  # [2, 100, 512, 512]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False, )
            processed_results = []

            for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                         mask_pred_results):
                image_size = x.shape[-2:]
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = self.sem_seg_postprocess(
                        mask_pred_result, image_size, image_size[0],
                        image_size[1])

                r = self.semantic_inference(mask_cls_result, mask_pred_result)

                if not self.sem_seg_postprocess_before_inference:
                    r = self.sem_seg_postprocess(r, image_size, image_size[0],
                                                 image_size[1])
                processed_results.append({"sem_seg": r})

            r = r[None, ...]
            return [r]

    def sem_seg_postprocess(self, result, img_size, output_height,
                            output_width):
        """
        Return semantic segmentation predictions in the original resolution.

        The input images are often resized when entering semantic segmentor. Moreover, in same
        cases, they also padded inside segmentor to be divisible by maximum network stride.
        As a result, we often need the predictions of the segmentor in a different
        resolution from its inputs.

        Args:
            result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
                where C is the number of classes, and H, W are the height and width of the prediction.
            img_size (tuple): image size that segmentor is taking as input.
            output_height, output_width: the desired output resolution.

        Returns:
            semantic segmentation prediction (Tensor): A tensor of the shape
                (C, output_height, output_width) that contains per-pixel soft predictions.
        """
        result = paddle.unsqueeze(result[:, :img_size[0], :img_size[1]], axis=0)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False)[0]
        return result

    def loss_computation(self, logits_list, losses, data):
        check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses['types'][i]
            coef_i = losses['coef'][i]

            loss_list.append(coef_i * loss_i(logits, data['instances']))
        return loss_list


class BasePixelDecoder(nn.Layer):
    def __init__(self, input_shape, conv_dim=256, norm="GN", mask_dim=256):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]['stride'])
        self.in_features = [k for k, v in input_shape]  # "res2" to "res5"
        feature_channels = [v['channels'] for k, v in input_shape]

        self.lateral_convs, self.output_convs = nn.LayerList(), nn.LayerList()
        use_bias = norm == ''
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_conv = layers.ConvNormAct(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    bias_attr=use_bias,
                    norm=nn.GroupNorm(
                        num_groups=32, num_channels=conv_dim),
                    act_type='relu')
                self.output_convs.append(output_conv)
                self.lateral_convs.append(None)
                for layer in output_conv.sublayers():
                    if hasattr(layer, "weight"):
                        param_init.kaiming_uniform(
                            layer.weight,
                            negative_slope=1,
                            nonlinearity='leaky_relu')
                    if getattr(layer, 'bias', None) is not None:
                        param_init.constant_init(layer.bias, value=0)
            else:
                lateral_norm = nn.GroupNorm(
                    num_groups=32, num_channels=conv_dim)
                output_norm = nn.GroupNorm(num_groups=32, num_channels=conv_dim)

                lateral_conv = layers.ConvNormAct(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias_attr=False,
                    norm=lateral_norm)
                output_conv = layers.ConvNormAct(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=use_bias,
                    norm=output_norm,
                    act_type='relu')
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

                for layer in output_conv.sublayers() + lateral_conv.sublayers():
                    if hasattr(layer, "weight"):
                        param_init.kaiming_uniform(
                            layer.weight,
                            negative_slope=1,
                            nonlinearity='leaky_relu')
                    if getattr(layer, 'bias', None) is not None:
                        param_init.constant_init(layer.bias, value=0)

        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]

        self.mask_features = layers.ConvNormAct(
            conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        for layer in self.mask_features.sublayers():
            if hasattr(layer, "weight"):
                param_init.kaiming_uniform(
                    layer.weight, negative_slope=1, nonlinearity='leaky_relu')
            if getattr(layer, 'bias', None) is not None:
                param_init.constant_init(layer.bias, value=0)

    def forward(self, features):
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]

            if lateral_conv is None:
                y = self.output_convs[idx](x)
            else:
                cur_fpn = self.lateral_convs[idx](x)
                y = cur_fpn + F.interpolate(
                    y, size=cur_fpn.shape[-2:], mode='nearest')
                y = self.output_convs[idx](y)
        return self.mask_features(y), None


class PositionEmbeddingSine(nn.Layer):
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be true is scale is not None")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = paddle.zeros(
                shape=(x.shape[0], x.shape[2], x.shape[3]), dtype='bool')
        not_mask = ~mask
        y_embed = paddle.cumsum(not_mask, axis=1, dtype='float32')
        x_embed = paddle.cumsum(not_mask, axis=2, dtype='float32')

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype='float32')
        dim_t = paddle.cast(dim_t, dtype='int64')
        tmp = paddle.ones_like(dim_t) * 2
        dim_t = self.temperature**(2 * paddle.floor_divide(dim_t, tmp) /
                                   self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = paddle.flatten(
            paddle.stack(
                (paddle.sin(pos_x[:, :, :, 0::2]),
                 paddle.cos(pos_x[:, :, :, 1::2])),
                axis=4),
            start_axis=3)
        pos_y = paddle.flatten(
            paddle.stack(
                (paddle.sin(pos_y[:, :, :, 0::2]),
                 paddle.cos(pos_y[:, :, :, 1::2])),
                axis=4),
            start_axis=3)
        pos = paddle.transpose(
            paddle.concat(
                (pos_y, pos_x), axis=3), perm=(0, 3, 1, 2))
        return pos


class EncoderLayer(nn.Layer):
    """
    The layer to compose the transformer encoder.
    
    Args:
        d_model(int): The input feature's channels.
        nhead(int): the number of head for MHSA.
        dim_feedforward(int): The internal channels of linear layer.
        dropout(int): the dropout probability.
        activation(str): the kind of activation that used.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.init_weight()

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def init_weight(self):
        param_init.multihead_fill(self.self_attn, True)
        param_init.th_linear_fill(self.linear1)
        param_init.th_linear_fill(self.linear2)

    def forward(self, src, src_mask, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        if src_key_padding_mask is not None:
            raise ValueError(
                "The multihead attention does not support key_padding mask, but got src_key_padding_mask is not None"
            )

        attn = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        src += self.dropout(attn)
        src = self.norm1(src)
        attn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src += self.dropout2(attn)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Layer):
    """
    The transformer encoder.
    
    Args:
        encoder_layer(nn.Layer): The base layer to compose the encoder.
        num_layers(int): How many layers is used in the encoder.
        norm(str): the kind of normalization that used before output.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.LayerList()
        for i in range(num_layers):
            self.layers.append(encoder_layer)
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src

        for layer in self.layers:
            # if pos is not none, all the encoder layer will have the position embedding
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.Layer):
    """
    The layer to compose the transformer decoder.
    
    Args:
        d_model(int): The input feature's channels.
        nhead(int): the number of head for MHSA.
        dim_feedforward(int): The internal channels of linear layer.
        dropout(int): the dropout probability.
        activation(str): the kind of activation that used.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiHeadAttention(
            d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def init_weight(self):
        param_init.multihead_fill(self.self_attn, True)
        param_init.multihead_fill(self.multihead_attn, True)
        param_init.th_linear_fill(self.linear1)
        param_init.th_linear_fill(self.linear2)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        if tgt_key_padding_mask or memory_key_padding_mask:
            raise ValueError(
                "The multihead attention does not support key_padding_mask")

        q = k = self.with_pos_embed(tgt, query_pos).transpose(perm=(
            1, 0, 2))  # [2, 100, 256]
        tgt = tgt.transpose(perm=(1, 0, 2))
        attn = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask).transpose(perm=(1, 0, 2))  # [100, 2, 256]
        tgt = tgt.transpose(perm=(1, 0, 2))  # [100, 2, 256]

        tgt += self.dropout1(attn)

        tgt = self.norm1(tgt)  # [100, 2, 256]
        q = self.with_pos_embed(tgt, query_pos).transpose(perm=(1, 0, 2))
        k = self.with_pos_embed(memory, pos).transpose(perm=(1, 0, 2))
        v = memory.transpose(perm=(1, 0, 2))
        attn = self.multihead_attn(
            query=q, key=k, value=v,
            attn_mask=memory_mask).transpose(perm=(1, 0, 2))
        tgt += self.dropout2(attn)
        tgt = self.norm2(tgt)  # [100, 2, 256]
        attn = self.linear2(
            self.dropout(self.activation(self.linear1(tgt))))  # [100, 2, 256]
        tgt += self.dropout3(attn)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Layer):
    """
    The transformer decoder.
    
    Args:
        encoder_layer(nn.Layer): The base layer to compose the decoder.
        num_layers(int): How many layers is used in the decoder.
        norm(str): the kind of normalization that used before output.
        return_intermediate(bool): Whether to output the intermediate feature.
    """

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=True):
        super().__init__()
        self.decoder_list = nn.LayerList()
        for i in range(num_layers):
            self.decoder_list.append(copy.deepcopy(decoder_layer))
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.decoder_list:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


class Transformer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                     activation)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                     activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.init_weight()

    def init_weight(self):
        for name, p in self.named_parameters():
            if len(p.shape) > 1 and ('attn' not in name):
                param_init.xavier_uniform(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC 
        bs, c, h, w = src.shape
        src = paddle.transpose(paddle.flatten(src, start_axis=2), (2, 0, 1))
        pos_embed = paddle.transpose(
            paddle.flatten(
                pos_embed, start_axis=2), (2, 0, 1))
        query_embed = paddle.stack([query_embed for i in range(bs)], axis=1)
        if mask is not None:
            mask = paddle.flatten(mask, start_axis=1)

        tgt = paddle.zeros_like(query_embed)  # No.querry, N, hdim [100, 2, 256]
        memory = self.encoder(
            src, src_key_padding_mask=mask,
            pos=pos_embed)  # HWxNxC memory = src
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed)

        return paddle.transpose(hs, (0, 2, 1, 3)), paddle.reshape(
            paddle.transpose(memory, (1, 2, 0)), (bs, c, h, w))


class MLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.init_weight()

    def init_weight(self):
        for layer in self.layers:
            param_init.th_linear_fill(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class TransformerPredictor(nn.Layer):
    def __init__(self,
                 in_channels,
                 mask_classification,
                 num_classes=150,
                 hidden_dim=256,
                 num_queries=100,
                 nheads=8,
                 dropout=0.1,
                 dim_feedforward=2048,
                 enc_layers=0,
                 dec_layers=6,
                 pre_norm=False,
                 deep_supervision=True,
                 mask_dim=256,
                 enforce_input_project=False):
        super().__init__()
        self.mask_classification = mask_classification
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2D(in_channels, hidden_dim, kernel_size=1)
            if hasattr(self.input_proj, "weight"):
                param_init.kaiming_uniform(
                    self.input_proj.weight,
                    negative_slope=1,
                    nonlinearity='leaky_relu')
            if getattr(self.input_proj, 'bias', None) is not None:
                param_init.constant_init(self.input_proj.bias, value=0)
        else:
            self.input_proj = nn.Sequential()

        self.aux_loss = deep_supervision

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.init_weight()

    def init_weight(self, ):
        param_init.th_linear_fill(self.class_embed)
        param_init.normal_init(self.query_embed.weight, mean=0.0, std=1.0)

    def forward(self, x, mask_features):
        pos = self.pe_layer(x)

        mask = None
        hs, memory = self.transformer(
            self.input_proj(x), mask, self.query_embed.weight, pos)

        out = {}
        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out["pred_logits"] = outputs_class[-1]

        if self.aux_loss:
            mask_embed = self.mask_embed(hs)

            output_seg_masks = paddle.einsum("lbqc,bchw->lbqhw", mask_embed,
                                             mask_features)
            out["pred_masks"] = output_seg_masks[-1]
            if self.mask_classification:
                out['aux_outputs'] = [{
                    "pred_logits": a,
                    "pred_masks": b
                } for a, b in zip(outputs_class[:-1], output_seg_masks[:-1])]
            else:
                out['aux_outputs'] = [{
                    "pred_masks": b
                } for b in output_seg_masks[:-1]]
        else:
            mask_embed = self.mask_embed(hs[-1])
            output_seg_masks = paddle.einsum("bqc,bchw->bqhw", mask_embed,
                                             mask_features)
            out["pred_masks"] = output_seg_masks

        return out


class MaskFormerHead(nn.Layer):
    def __init__(self, input_shape, num_classes, transformer_in_feature='res5'):
        super(MaskFormerHead, self).__init__()
        self.transformer_in_feature = transformer_in_feature
        self.input_shape = input_shape
        self.pixel_decoder = BasePixelDecoder(input_shape)
        self.predictor = TransformerPredictor(
            input_shape[transformer_in_feature]["channels"],
            mask_classification=True,
            num_classes=num_classes)

    def forward(self, x):
        mask_features, transformer_encoder_features = self.pixel_decoder(x)
        predictions = self.predictor(x[self.transformer_in_feature],
                                     mask_features)

        return predictions
