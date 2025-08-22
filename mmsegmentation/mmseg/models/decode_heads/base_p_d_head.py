# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import PModule, DModule, Bag
import torch.nn as nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

@MODELS.register_module()
class BaselinePDHead(BaseDecodeHead):
    """Baseline + P + D head (PIDNet) for mapping feature to a predefined set
    of classes.

    Args:
        in_channels (int): Number of feature maps coming from 
        the decoded prediction.
            Default: 256.
        num_classes (int): Number of classes in the training
        dataset.
            Default: 19 for Cityscapes.
    """

    def __init__(self, in_channels=256, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(in_channels, int)
        assert isinstance(num_classes, int)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.p_module = PModule(channels=self.in_channels // 4)
        self.d_module = DModule(channels=self.in_channels // 4)
        self.fusion = Bag(self.in_channels, self.in_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg_dfm)
        if self.training:
            self.p_head = nn.Conv2d(self.in_channels // 2, self.num_classes, kernel_size=1)
            self.d_head = nn.Conv2d(self.in_channels // 2, 1, kernel_size=1)
        self.seg_head = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward function.
        x should be a tuple of outputs:
        x_2, x_3, x_4, x_out = x
        x_2 has shape (N, 128, H/8, W/8)
        x_3 has shape (N, 256, H/16, W/16)
        x_4 has shape (N, 512, H/32, W/32)
        x_out has shape (N, 256, H/8, W/8)
        """
        if self.training:
            temp_p, x_p = self.p_module(x) # temp_p: (N, 128, H/8, W/8), x_p: (N, 256, H/8, W/8)
            temp_d, x_d = self.d_module(x) # temp_d: (N, 128, H/8, W/8), x_d: (N, 256, H/8, W/8)
            p_supervised = self.p_head(temp_p)
            d_supervised = self.d_head(temp_d)
            feats = self.fusion(x_p, x[3], x_d)
            output = self.seg_head(feats)
            return tuple([output, p_supervised, d_supervised])
        else:
            x_p = self.p_module(x)
            x_d = self.d_module(x)
            feats = self.fusion(x_p, x[3], x_d)
            output = self.seg_head(feats)
            return output