from typing import Optional, Dict
import torch
from torch.optim import Optimizer
from mmengine.optim import OptimWrapper
from mmpretrain.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPERS.register_module()
class GradTrackingOptimWrapper(OptimWrapper):
    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        super().__init__(optimizer, accumulative_counts, clip_grad)
        self.runner = None # Will be set automatically by the runner during initialization
        
    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)