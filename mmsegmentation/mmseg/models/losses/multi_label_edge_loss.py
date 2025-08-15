#!/usr/bin/env python3

import torch
import torch.nn as nn

from .binary_edge_loss import balanced_binary_loss, weighted_binary_loss
from ..builder import LOSSES

'''
@LOSSES.register_module()
class MultiLabelEdgeLoss(nn.Module):
    """Class Balanced Multilabel Loss used in DFF"""

    def __init__(
        self,
        loss_weight=1.0,
        loss_name="loss_multilabel_edge",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        **kwargs,
    ):
        loss_total = 0

        # FIXME: could optimize for batched loss
        for i in range(edge_label.size(0)):  # iterate for batch size
            pred = edge[i]
            target = edge_label[i]

            num_pos = torch.sum(target)  # true positive number
            num_total = target.size(-1) * target.size(-2)  # true total number
            num_neg = num_total - num_pos
            # compute a pos_weight for each image
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total)

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = (
                pred
                - pred * target
                + log_weight
                * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())
            )

            loss = loss.mean()
            loss_total = loss_total + loss

        loss_total = loss_total / edge_label.size(0)
        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name
'''

@LOSSES.register_module()
class MultiLabelEdgeLoss(nn.Module):
    """Class Balanced Multilabel Loss used in DFF"""
    def __init__(
        self,
        loss_weight=1.0,
        loss_name="loss_multilabel_edge",
        ignore_index=255,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index
    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        **kwargs,
    ):
        loss_total = 0
        # FIXME: could optimize for batched loss
        for i in range(edge_label.size(0)):  # iterate for batch size
            pred = edge[i]
            target = edge_label[i]
            valid_mask = (target != self.ignore_index)
            target_valid = target.clone()
            target_valid[~valid_mask] = 0  # Set ignore to 0 for computation
            num_pos = (target == 1).float().sum(dim=[1, 2])
            num_neg = (target == 0).float().sum(dim=[1, 2])
            num_total = num_pos + num_neg
            # compute a pos_weight for each class in the image
            pos_weight = (num_neg / num_pos.clamp(min=1e-6)).clamp(min=torch.ones_like(num_total), max=num_total)
            pos_weight = pos_weight.unsqueeze(1).unsqueeze(2)
            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target_valid
            loss = (
                pred
            - pred * target_valid
            + log_weight
            * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())
                )

            # Mask the loss to ignore invalid pixels
            loss = loss[valid_mask]
            if loss.numel() > 0:
                loss = loss.mean()
            else:
                loss = pred.new_tensor(0.0)
            loss_total = loss_total + loss
        loss_total = loss_total / edge_label.size(0)
        return self.loss_weight * loss_total
    
    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class BalancedMultiLabelLoss(nn.Module):
    def __init__(
        self,
        sensitivity=10,
        loss_weight=1.0,
        loss_name="loss_balanced_multilabel_edge",
    ):
        super().__init__()
        self.sensitivity = sensitivity
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        loss_total = 0

        for i in range(edge_label.size(1)):  # iterate for classes
            pred = edge[:, i, ...].unsqueeze(1)
            target = edge_label[:, i, ...].unsqueeze(1)

            loss = balanced_binary_loss(
                edge=pred,
                edge_label=target,
                sensitivity=self.sensitivity,
                ignore_index=ignore_index,
                reduction="mean",
            )
            loss_total = loss_total + loss

        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class WeightedMultiLabelLoss(nn.Module):
    """Weighted Multi-label Loss

    Weighing `alpha=beta` results in high recall,
    so reducing `beta` by `1/num_classes` stabilizes.
    """

    def __init__(
        self,
        num_classes=19,
        loss_weight=1.0,
        loss_name="loss_multilabel_bce_edge",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self._alpha = 1.0
        self._beta = float(num_classes)

    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        loss_total = 0

        for i in range(edge_label.size(1)):  # iterate for classes
            pred = edge[:, i, ...].unsqueeze(1)
            target = edge_label[:, i, ...].unsqueeze(1)

            loss = weighted_binary_loss(
                edge=pred,
                edge_label=target,
                alpha=self._alpha,
                beta=self._beta,
                ignore_index=ignore_index,
                reduction="mean",
            )
            loss_total = loss_total + loss

        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name