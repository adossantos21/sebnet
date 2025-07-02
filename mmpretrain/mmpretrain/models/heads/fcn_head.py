# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from .cls_head import ClsHead
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class FCNHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels_main: int,
                 in_channels_aux: int,
                 loss1: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss2: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(FCNHead, self).__init__(init_cfg=init_cfg, **kwargs)
        if not isinstance(loss1, nn.Module):
            loss1 = MODELS.build(loss1)
        if not isinstance(loss2, nn.Module):
            loss2 = MODELS.build(loss2)
        self.loss_module_1 = loss1
        self.loss_module_2 = loss2
        self.in_channels_main = in_channels_main
        self.in_channels_aux = in_channels_aux
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc1 = nn.Linear(self.in_channels_main, self.num_classes)
        self.fc2 = nn.Linear(self.in_channels_aux, self.num_classes)
    
    def forward(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """The forward process."""
        feats_aux, feats_main = feats
        # The final classification head.
        cls_score_1 = self.fc1(feats_main)
        cls_score_2 = self.fc2(feats_aux)
        return (cls_score_1, cls_score_2)

    def _get_loss(self, cls_score: Tuple[torch.Tensor],
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        cls_score_1, cls_score_2 = cls_score
        losses = dict()
        loss1 = self.loss_module_1(
            cls_score_1, target, avg_factor=cls_score_1.size(0), **kwargs)
        loss2 = self.loss_module_2(
            cls_score_2, target, avg_factor=cls_score_2.size(0), **kwargs)
        
        losses['loss_main'] = loss1
        losses['loss_aux'] = loss2

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score_1, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses