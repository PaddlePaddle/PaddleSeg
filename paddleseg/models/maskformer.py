# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import math
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils


class BasePixelDecoder(nn.Layer):
    def __init__(self, input_shape, conv_dim=256, norm="GN", mask_dim=256):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]['stride'])
        self.in_features = [k for k, v in input_shape
                            ]  # starting from "res2" to "res5"
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
                # self.add_sublayer("layer_{}".format(idx + 1), output_conv)
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
                # self.add_sublayer('adapter_{}'.format(idx + 1), lateral_conv)
                # self.add_sublayer('layer_{}'.format(idx + 1), output_conv)
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]

        self.mask_features = layers.ConvNormAct(
            conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)

    def __init_weight__(self, ):
        for layer in self.sublayers():
            param_init.xavier_uniform(layer.weight)
            if layer.bias is not None:
                param_init.xavier_uniform(layer.bias)

            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

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


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

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


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

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

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,  # forward post
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

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                param_init.xavier_uniform(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC 
        bs, c, h, w = src.shape
        src = paddle.transpose(paddle.flatten(src, start_axis=2), (2, 0, 1))
        pos_embed = paddle.transpose(
            paddle.flatten(
                pos_embed, start_axis=2), (2, 0, 1))
        query_embed = paddle.repeat_interleave(
            paddle.unsqueeze(
                query_embed, axis=1), repeats=bs, axis=1)  # Noquerry, N, hdim
        if mask is not None:
            mask = paddle.flatten(mask, start_axis=1)

        tgt = paddle.zeros_like(query_embed)  # Noquerry, N, hdim [100, 2, 256]
        memory = self.encoder(
            src, src_key_padding_mask=mask,
            pos=pos_embed)  # HWxNxC 中间输出memory = src
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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class TransformerPredictor(nn.Layer):
    def __init__(
            self,
            in_channels,
            mask_classification,
            num_classes=150,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dropout=0.1,  # TODO change to 0.1
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

        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # lookup dict

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2D(in_channels, hidden_dim, kernel_size=1)
            param_init.xavier_uniform(self.input_proj.weight)
            param_init.xavier_uniform(self.input_proj.bias)
        else:
            self.input_proj = nn.Sequential()

        self.aux_loss = deep_supervision

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features):
        pos = self.pe_layer(x)

        mask = None
        hs, memory = self.transformer(
            self.input_proj(x), mask, self.query_embed.weight, pos)

        out = {}
        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out["pred_logits"] = outputs_class[-1]  # done align

        if self.aux_loss:
            mask_embed = self.mask_embed(hs)

            output_seg_masks = paddle.einsum("lbqc,bchw->lbqhw", mask_embed,
                                             mask_features)
            out["pred_masks"] = output_seg_masks[-1]  # done align
            if self.mask_classification:
                out['aux_outputs'] = [
                    {  # done align
                        "pred_logits": a,
                        "pred_masks": b
                    } for a, b in zip(outputs_class[:-1], output_seg_masks[:-1])
                ]
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

    def forward(self, x):  # {"res2": xx, "res3": xx, "res4": xx, "res5": xx}
        mask_features, transformer_encoder_features = self.pixel_decoder(x)
        predictions = self.predictor(x[self.transformer_in_feature],
                                     mask_features)

        return predictions


def sem_seg_postprocess(result, img_size, output_height, output_width):
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


@manager.MODELS.add_component
class MaskFormer(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 size_divisibility=32,
                 sem_seg_postprocess_before_inference=False,
                 pretrained=None):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
        """
        super(MaskFormer, self).__init__()
        self.num_classes = num_classes
        self.size_divisibility = size_divisibility
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

    def forward(self, x, test=False):
        if test:
            import numpy as np
            np.random.seed(0)
            a = np.random.random([2, 3, 512, 512]) * 0.7620
            x = paddle.to_tensor(a, dtype='float32')
            # x = paddle.ones([2, 3, 512, 512]) * 0.7620

        features = self.backbone(x)
        outputs = self.seghead(features)

        if self.training:
            return [outputs]
        else:  # done 
            mask_cls_results = outputs["pred_logits"]  # [2, 100, 151]
            mask_pred_results = outputs["pred_masks"]  # # [2, 100, 512, 512]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False, )
            processed_results = []  #TODO can we change slice to pack here?

            for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                         mask_pred_results):

                image_size = x.shape[-2:]

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, image_size[0],
                        image_size[1])

                # semantic segmentation inference
                r = self.semantic_inference(mask_cls_result,
                                            mask_pred_result)  # done

                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, image_size[0],
                                            image_size[1])
                processed_results.append({"sem_seg": r})

            r = r[None, ...]
            return [r]


if __name__ == "__main__":
    from paddleseg.models.backbones import SwinTransformer_tiny_patch4_window7_384
    backbone = SwinTransformer_tiny_patch4_window7_384()
    model = MaskFormer(
        backbone=backbone,
        num_classes=150,
        pretrained="saved_model/maskformer_tiny.pdparams")

    # with open('maskformer_tiny_paddle_model.txt', 'w') as f:
    #     for keys, values in model.named_parameters():
    #         f.write(keys +'\t'+str(values.shape)+'\t'+str(values.mean())+"\n")
    x = paddle.ones([2, 3, 512, 512]) * 0.7620
    out = model(x)
