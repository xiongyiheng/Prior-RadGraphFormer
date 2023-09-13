# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .utils import NestedTensor, is_main_process
from .position_encoding_2D import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, name: str):
        super().__init__()

        for backbone_name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        if return_interm_layers:
            if name[0:8] == "densenet":
                return_layers = {"denseblock2": "0", "denseblock3": "1", "denseblock4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 1024]
            else:
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                # self.strides = [4, 8, 16, 32]
                # self.num_channels = [256, 512, 1024, 2048]
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
        else:
            if name[0:8] == "densenet":
                return_layers = {'denseblock4': "0"}
                self.strides = [32]
                self.num_channels = [1024]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers) if name[0:8] == "densenet" \
            else IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name[0:8] == "densenet":
            backbone = getattr(torchvision.models, name)(pretrained=False)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=False, norm_layer=norm_layer)  # hard code pretrained to false
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers, name)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(config):
    position_embedding = build_position_encoding(config)
    train_backbone = float(config.MODEL.ENCODER.LR_BACKBONE) > 0
    return_interm_layers = config.MODEL.ENCODER.MASKS or (config.MODEL.ENCODER.NUM_FEATURE_LEVELS > 1)
    backbone = Backbone(
        name=config.MODEL.ENCODER.BACKBONE, train_backbone=train_backbone, return_interm_layers=return_interm_layers,
        dilation=config.MODEL.ENCODER.DILATION
    )
    model = Joiner(backbone, position_embedding)
    return model
