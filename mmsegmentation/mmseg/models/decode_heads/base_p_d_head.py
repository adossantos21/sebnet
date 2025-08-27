# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import BaseSegHead, PModule, Bag
#from mmseg.models.utils import DModule
from mmseg.models.utils import DModule_EarlierLayers as DModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from typing import List, Tuple
from mmseg.utils import OptConfigType, SampleList
from torch import Tensor

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

    def __init__(self, 
                 in_channels: int = 256, 
                 num_classes: int = 19, 
                 num_stem_blocks: int = 3,
                 norm_cfg: OptConfigType = dict(type='SyncBN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            in_channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        assert isinstance(in_channels, int)
        assert isinstance(num_classes, int)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stem_blocks = num_stem_blocks
        self.stride = 1
        self.p_module = PModule(channels=self.in_channels // 4, num_stem_blocks=self.num_stem_blocks)
        self.d_module = DModule(channels=self.in_channels // 4, num_stem_blocks=self.num_stem_blocks)
        self.fusion = Bag(self.in_channels, self.in_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        if self.training:
            self.p_head = BaseSegHead(self.in_channels // 2, self.in_channels, self.stride, norm_cfg, act_cfg)
            self.d_head = BaseSegHead(self.in_channels // 2, self.in_channels // 4, self.stride, norm_cfg)
            self.p_cls_seg = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)
            self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        self.seg_head = BaseSegHead(self.in_channels, self.in_channels, self.stride, norm_cfg, act_cfg)

    def forward(self, x):
        """
        Forward function.
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        """
        if self.training:
            temp_p, x_p = self.p_module(x) # temp_p: (N, 128, H/8, W/8), x_p: (N, 256, H/8, W/8)
            temp_d, x_d = self.d_module(x) # temp_d: (N, 128, H/8, W/8), x_d: (N, 256, H/8, W/8)
            p_supervised = self.p_head(temp_p, self.p_cls_seg) # (N, K, H/8, W/8), where K is the number of classes in the labeled dataset
            d_supervised = self.d_head(temp_d, self.d_cls_seg) # (N, 1, H/8, W/8)
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            feats = self.fusion(x_p, x[-1], x_d)
            output = self.seg_head(feats, self.cls_seg) # (N, K, H/8, W/8)
            return tuple([output, p_supervised, d_supervised])
        else:
            x_p = self.p_module(x)
            x_d = self.d_module(x)
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            feats = self.fusion(x_p, x[-1], x_d)
            output = self.seg_head(feats, self.cls_seg)
            return output
        
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        i_logit, p_logit, d_logit = seg_logits
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        p_logit = resize(
            input=p_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        i_logit = resize(
            input=i_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        d_logit = resize(
            input=d_logit,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)
        loss['loss_sem_p'] = self.loss_decode[0](
            p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        loss['loss_bd'] = self.loss_decode[2](d_logit, bd_label)
        loss['acc_seg'] = accuracy(
            i_logit, sem_label, ignore_index=self.ignore_index)
        
        logits = dict(
            seg_logits=i_logit,
            p_logits=p_logit,
            d_logits=d_logit
        )
        return loss, logits