from mmengine.runner import EpochBasedTrainLoop
from mmpretrain.registry import LOOPS
from typing import Sequence

@LOOPS.register_module()
class GradientTrackingEpochTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        
        # update_params is where self.backward() is called. So the hook needs to be called after self.backward()
        self.runner.call_hook(
            'after_backward_pass', batch_idx=idx, data_batch=data_batch)
        
        if self.runner.optim_wrapper.should_update():
            self.runner.optim_wrapper.step()
            self.runner.optim_wrapper.zero_grad()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1