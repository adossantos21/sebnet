import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.models.losses import accuracy
from mmseg.models.utils import (
    resize,
    BaseSegHead,
    EdgeModuleConditioned as EdgeModule
)
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList
from .decode_head import BaseDecodeHead
from typing import Tuple

@MODELS.register_module()
class Ablation05(BaseDecodeHead):
    """
    Ablation 05 - Baseline + Edge Head, conditioned with a HED supervisory signal.
    No fusion. See Holistically-Nested Edge Detection (HED) at
    https://arxiv.org/pdf/1504.06375.pdf for more details.

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
                 stride: int = 1,
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
        assert isinstance(in_channels, int), f"Expected in_channels to be int, got {type(in_channels)}"
        assert isinstance(num_classes, int), f"Expected num_classes to be int, got {type(num_classes)}"
        assert isinstance(num_stem_blocks, int), f"Expected num_stem_blocks to be int, got {type(num_stem_blocks)}"
        assert isinstance(stride, int), f"Expected stride to be int, got {type(stride)}"
        self.eval_edges = eval_edges
        if self.training or self.eval_edges:
            self.edge_module = EdgeModule(channels=in_channels // 4, num_stem_blocks=num_stem_blocks, eval_edges=self.eval_edges)
            self.hed_head = BaseSegHead(in_channels // 2, in_channels // 4, stride=stride, norm_cfg=norm_cfg) # No act_cfg here on purpose. See pidnet head.
            self.hed_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        self.seg_head = BaseSegHead(in_channels, in_channels, stride=stride, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: Tuple[Tensor, ...]):
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
            x_edges = self.edge_module(x) # x_edges: (N, 128, H/8, W/8)
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            hed = self.hed_head(x_edges, self.hed_cls_seg) # hed: (N, 1, H/8, W/8)
            output = self.seg_head(x[-1], self.cls_seg) # output: (N, K, H/8, W/8) where K = num_classes
            return tuple([output, hed])
        else:
            if self.eval_edges:
                x_edges = self.edge_module(x)
                hed = self.hed_head(x_edges, self.hed_cls_seg)
                output = tuple([hed])
            else:
                x[-1] = F.interpolate(
                    x[-1],
                    size=x[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                )
                output = self.seg_head(x[-1], self.cls_seg)
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

    def loss_by_feat(self, logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        seg_logits, hed_logits = logits
        seg_label, hed_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        hed_logits = resize(
            input=hed_logits,
            size=hed_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        hed_label = hed_label.squeeze(1)
        logits = dict(
            seg_logits=seg_logits,
            hed_logits=hed_logits,
        )
        loss = dict()
        loss['loss_seg'] = self.loss_decode[0](seg_logits, seg_label)
        loss['loss_hed'] = self.loss_decode[1](hed_logits, hed_label)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss, logits