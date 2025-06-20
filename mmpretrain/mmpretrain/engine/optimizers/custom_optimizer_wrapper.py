from typing import Optional
from torch.optim import Optimizer
from mmengine.optim import OptimWrapper
from mmpretrain.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPERS.register_module()
class SuperCustomOptimWrapper(OptimWrapper):
    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        super().__init__(optimizer, accumulative_counts, clip_grad)
        self.runner = None # Will be set automatically by the runner during initialization

    def backward(self, loss, **kwargs):
        """
        Override backward to call hook after gradient computation, so that we can 
        visualize gradients.
        """
        super().backward(loss, **kwargs)

        # Call custom hook after backward pass
        if self.runner is not None:
            self.runner.call_hook('after_backward_pass')
        else:
            raise RuntimeError(f"self.runner is None")
