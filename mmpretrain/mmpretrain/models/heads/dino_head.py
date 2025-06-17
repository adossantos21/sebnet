# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from .cls_head import ClsHead
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from typing import List, Optional, Tuple, Union

import sys

@MODELS.register_module()
class DINOHead(ClsHead): # Changed from nn.Module to ClsHead, which inherits from BaseModule, which inherits from nn.Module
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is a tuple with two items. The first are the features with spatial dimensions. The second has the spatial dimensions collapsed.
        # x[0] has shape (32, 1024, 4, 4)
        x = x[0]
        x = self.gap(x)
        x = torch.flatten(x, 1) # shape (32, 1024)
        #print(f"gap shape: {x.shape}")
        x = self.mlp(x)
        #print(f"mlp shape: {x.shape}")
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        # The final classification head
        x = self.last_layer(x)
        #print(f"output shape: {x.shape}")
        return x

    def loss(self, feats:torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> dict:
        # self(feats) invokes __call__ method of nn.Linear(), which invokes the forward() method
        # Thus, cls_score is equal to what is returned from the forward() method, i.e., x.
        cls_score = self(feats) 
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)