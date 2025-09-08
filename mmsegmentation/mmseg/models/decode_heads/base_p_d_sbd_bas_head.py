# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import BaseSegHead, Bag, PModule, DModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from typing import Tuple
from mmseg.utils import OptConfigType, SampleList
from torch import Tensor

@MODELS.register_module()
class BaselinePDSBDBASHead(BaseDecodeHead):
    """Baseline + P Branch + D Branch + SBD head for mapping feature to a predefined set
    of classes (with bag fusion).

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
                 eval_edges: bool = False,
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
        self.stride = 1
        self.num_stem_blocks = num_stem_blocks
        self.eval_edges = eval_edges
        self.p_module = PModule(channels=self.in_channels // 4, num_stem_blocks=self.num_stem_blocks)
        self.p_head = BaseSegHead(self.in_channels // 2, self.in_channels, self.stride, norm_cfg, act_cfg)
        self.p_cls_seg = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)
        self.fusion = Bag(self.in_channels, self.in_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.seg_head = BaseSegHead(self.in_channels, self.in_channels, self.stride, norm_cfg, act_cfg)
        self.sbd = DModule(channels=self.in_channels // 4, num_stem_blocks=self.num_stem_blocks, eval_edges=self.eval_edges)
        self.d_head = BaseSegHead(self.in_channels // 2, self.in_channels // 4, self.stride, norm_cfg) # No act_cfg here on purpose. See pidnet head.
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        self.sbd_head = BaseSegHead(self.in_channels // 2, self.in_channels // 4, self.stride, norm_cfg) # No act_cfg here on purpose. See pidnet head.
        self.sbd_cls_seg = nn.Conv2d(in_channels // 4, self.num_classes, kernel_size=1)

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
            temp_p, x_p = self.p_module(x)
            p_supervised = self.p_head(temp_p, self.p_cls_seg)
            temp_d, x_d = self.sbd(x)
            d_supervised = self.d_head(temp_d, self.d_cls_seg)
            sbd_supervised = self.sbd_head(temp_d, self.sbd_cls_seg)
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            feats = self.fusion(x_p, x[-1], x_d)
            output = self.seg_head(feats, self.cls_seg)
            return tuple([output, p_supervised, d_supervised, sbd_supervised])
        else:
            if self.eval_edges:
                temp_d, _ = self.sbd(x)
                d_output = self.d_head(temp_d, self.d_cls_seg)
                sbd_output = self.sbd_head(temp_d, self.sbd_cls_seg)
                output = tuple([d_output, sbd_output])
            else:
                x_p = self.p_module(x)
                x_d = self.sbd(x)
                x[-1] = F.interpolate(
                    x[-1],
                    size=x[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
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
        gt_multi_edge_segs = [
            data_sample.gt_multi_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        gt_multi_edge_segs = torch.stack(gt_multi_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs, gt_multi_edge_segs

    def loss_by_feat(self, logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        seg_logits, p_logits, d_logits, sbd_logits = logits
        seg_label, bd_label, bd_multi_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        p_logits = resize(
            input=p_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        d_logits = resize(
            input=d_logits,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sbd_logits = resize(
            input=sbd_logits,
            size=bd_multi_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        bd_label = bd_label.squeeze(1)
        bd_multi_label = bd_multi_label.squeeze(1)
        logits = dict(
            seg_logits=seg_logits,
            p_logits=p_logits,
            d_logits=d_logits,
            sbd_logits=sbd_logits
        )
        loss = dict()
        loss['loss_seg'] = self.loss_decode[0](seg_logits, seg_label)
        loss['loss_seg_p'] = self.loss_decode[1](p_logits, seg_label)
        loss['loss_bd'] = self.loss_decode[2](d_logits, bd_label)
        loss['loss_sbd'] = self.loss_decode[3](sbd_logits, bd_multi_label)
        filler = torch.ones_like(seg_label) * self.ignore_index
        sem_bd_label = torch.where(
            torch.sigmoid(d_logits[:, 0, :, :]) > 0.8, seg_label, filler)
        loss['loss_bas'] = self.loss_decode[4](seg_logits, sem_bd_label)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss, logits