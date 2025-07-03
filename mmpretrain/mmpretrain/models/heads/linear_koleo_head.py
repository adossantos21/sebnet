# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class LinearKoLeoHead(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss1: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss2: dict = dict(type='KoLeoLoss', loss_weight=0.1),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearKoLeoHead, self).__init__(init_cfg=init_cfg, **kwargs)
        if not isinstance(loss1, nn.Module):
            loss1 = MODELS.build(loss1)
        if not isinstance(loss2, nn.Module):
            loss2 = MODELS.build(loss2)
        self.loss_module_1 = loss1
        self.loss_module_2 = loss2
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cal_acc = cal_acc

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
    
    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor]:
        """The forward process."""
        if self.training:
            # The final classification head.
            cls_score = self.fc(feats)
            return (cls_score, feats)
        else:
            return self.fc(feats)
    
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                The shape of every item should be ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: Tuple[torch.Tensor],
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        cls_score, feats = cls_score
        losses = dict()
        loss1 = self.loss_module_1(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        loss2 = self.loss_module_2(feats, eps=1e-8, **kwargs)
        
        losses['loss_main'] = loss1
        losses['loss_koleo'] = loss2

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses
    
    def predict(
        self,
        feats: torch.Tensor,
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (torch.Tensor): The features extracted from the backbone.
                The shape should be ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples