"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

import math
import copy

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid, reset_parameters

from models.schemata.assimilation import Assimilation


class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs RadGraph generation """

    def __init__(self, backbone, transformer, config, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.config = config

        self.num_queries = config.MODEL.DECODER.NUM_QUERIES
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.focal_loss = not config.TRAIN.FOCAL_LOSS_ALPHA == ''
        self.num_label_classes = config.MODEL.NUM_LABEL_CLS
        self.num_token_classes = config.MODEL.NUM_TOKEN_CLS + 1

        self.label_embed = nn.Linear(self.hidden_dim, self.num_label_classes)
        self.token_embed = nn.Linear(self.hidden_dim, self.num_token_classes)

        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL  # add additional embedding in relation
        self.use_dropout = config.MODEL.DECODER.DROPOUT_REL
        if self.use_dropout:  # this dropout sub be used only for obj and freq bias to force rel token to learn
            self.dropout_rel = nn.Dropout(p=config.MODEL.DECODER.DROPOUT)

        # relation embedding
        num_of_token = 3
        input_dim = (self.hidden_dim * num_of_token + 8 + 2 * self.num_token_classes) if self.add_emd_rel \
            else self.hidden_dim * num_of_token  # +2*self.num_classes # need to check
        feed_fwd = config.MODEL.DECODER.DIM_FEEDFORWARD if hasattr(config.MODEL.DECODER,
                                                                   'NORM_REL_EMB') and config.MODEL.DECODER.NORM_REL_EMB else self.hidden_dim
        self.asm = Assimilation(in_edge_dim=self.hidden_dim,
                                hidden_edge_dim=self.hidden_dim,
                                out_edge_dim=self.hidden_dim,
                                in_node_dim=self.hidden_dim,
                                hidden_node_dim=self.hidden_dim,
                                num_heads=5,
                                n_edge_class=config.MODEL.NUM_REL_CLS + 1,
                                n_node_class=self.num_token_classes,
                                asm_num=2,
                                freeze_base=False,
                                yesFuse=True, hard_att=False, sigmoid_uncertainty=False).cuda()  # some are hard-coded
        self.project = MLP(input_dim, feed_fwd, self.hidden_dim, 3,
                           use_norm=hasattr(config.MODEL.DECODER, 'NORM_REL_EMB') and config.MODEL.DECODER.NORM_REL_EMB,
                           dropout=config.MODEL.DECODER.DROPOUT)

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        if config.TRAIN.FOCAL_LOSS_ALPHA:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.label_embed.bias.data = torch.ones(self.num_label_classes) * bias_value
            self.token_embed.bias.data = torch.ones(self.num_token_classes) * bias_value
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])

        # Deformable Transformer backbone
        features, pos = self.backbone(samples)

        # Create 
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight

        hs, init_reference, inter_references, attn_map, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, query_embeds, pos
        )

        object_token = hs[..., :-1, :]
        outputs_token = self.token_embed(object_token)
        outputs_label = self.label_embed(object_token)

        if self.aux_loss:  # with intermediate layer and aux_loss
            out = {'pred_token_logits': outputs_token[-1], 'pred_label_logits': outputs_label[-1],
                   'aux_outputs': self._set_aux_loss(outputs_token, outputs_label)}  # -1 means the last decoder layer

        else:
            out = {'pred_token_logits': outputs_token, 'pred_label_logits': outputs_label}

        cls_emb = self.dropout_rel(hs[..., :-1, :]) if self.use_dropout else hs[..., :-1, :]  # for both label and token
        rel_emb = hs[..., -1, :]  # the last one is for relation
        out['attn_map'] = attn_map

        if self.add_emd_rel:
            cls_emb = torch.cat((cls_emb, outputs_token, outputs_label), -1)
        return (cls_emb, rel_emb), out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_token, outputs_label):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_token_logits': a, 'pred_label_logits': b}
                for a, b in zip(outputs_token[:-1], outputs_label[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_norm=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.use_norm = False
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if use_norm:
            self.use_norm = True
            self.layers.insert(0, nn.LayerNorm(input_dim))
            # self.layers.insert(2,nn.Dropout(p=dropout))
            # self.layers.insert(4, nn.Dropout(p=dropout))
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, retun_interm=False):
        if self.use_norm:
            x = self.layers[3](self.dropout2(F.relu(self.layers[2](self.dropout1(self.layers[1](self.layers[0](x)))))))
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                # if  retun_interm and i==self.num_layers-2 :
                #     interm_feats = x
            # if retun_interm:
            #     return x,interm_feats
            # else:
            #
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_relationformer(config, **kwargs):
    if 'swin' in config.MODEL.ENCODER.BACKBONE:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(config.MODEL.ENCODER.BACKBONE)
    else:
        backbone = build_backbone(config)
    transformer = build_deforamble_transformer(config)

    model = RelationFormer(
        backbone,
        transformer,
        config,
        **kwargs
    )

    return model
